"""
GateVision → Prosper Smart Yard API (gate events).

POST /api/sites/{siteId}/gate-events
Body: CreateGateEventRequest (see syapi Swagger).
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.app_logger import get_logger
from app.container_utils import (
    _clean_trailer_number,
    _extract_trailer_and_scac,
    _iso_timestamp,
    _is_uuid,
    _post_with_auth_retry,
    prosper_device_uuid,
    prosper_event_uuid,
)

log = get_logger(__name__)

_GATE_UUID_NS = uuid.UUID("048e5f3e-7b3a-7f3a-8b1c-0e7d1a2b3c4d")


def is_gatevision_test_recording_row(record: Dict[str, Any]) -> bool:
    """
    True for offline recording test rows (metrics GateVision test pipeline).

    These use video_path gatevision:test-<stem>:gate_pass so Prosper gate-events
    can ingest them; local crop images are kept after upload (see main app).
    """
    vp = (record.get("video_path") or "").strip().lower()
    return vp.startswith("gatevision:test-") and vp.endswith(":gate_pass")


def parse_gatevision_video_path(video_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Parse ``gatevision:{gate_id}:{stage}`` into (gate_id, stage)."""
    s = (video_path or "").strip()
    low = s.lower()
    if not low.startswith("gatevision:"):
        return None, None
    rest = s[len("gatevision:") :]
    idx = rest.rfind(":")
    if idx <= 0:
        return None, None
    gate_id = rest[:idx].strip()
    stage = rest[idx + 1 :].strip()
    return (gate_id or None, stage or None)


def prosper_gate_uuid(local_gate_id: str, gate_id_map: Optional[Dict[str, str]] = None) -> str:
    """Map configured gate id (e.g. gate-1) to Prosper gate UUID."""
    raw = (local_gate_id or "").strip()
    if not raw:
        raw = "gate-unknown"
    if gate_id_map:
        mapped = gate_id_map.get(raw)
        if mapped is None:
            mapped = gate_id_map.get(raw.lower())
        if mapped and _is_uuid(str(mapped)):
            return str(uuid.UUID(str(mapped).strip())).lower()
    if _is_uuid(raw):
        return str(uuid.UUID(raw)).lower()
    return str(uuid.uuid5(_GATE_UUID_NS, raw)).lower()


def stage_to_gate_event_type(stage: str) -> Optional[int]:
    """
    Map fused GateVision stage to Prosper GateEventType (Swagger enum: 1, 2).

    Convention aligned with arrival / departure semantics:
      gate_arrival → 1, gate_departure → 2, gate_pass → 1 (unknown direction).
    """
    s = (stage or "").strip().lower()
    if s == "gate_arrival":
        return 1
    if s == "gate_departure":
        return 2
    if s == "gate_pass":
        return 1
    return None





def _resolve_gate_local_for_upload(
    record: Dict[str, Any],
    gate_local_from_path: str,
    gate_id_map: Optional[Dict[str, str]],
) -> str:
    """Return the local gate-id string to look up in ``prosper_gate_id_map``.

    For LIVE gate-pass rows the DB row has ``gate_id="gate-1"`` (or similar),
    legacy reconstruction makes ``gatevision:gate-1:gate_pass``, and the
    parser hands us ``gate_local="gate-1"`` — which is already what the map
    is keyed on, so we just return it unchanged.

    For TEST recordings the row has ``gate_id=NULL`` and
    ``test_video_stem=<long_video_stem>``; legacy reconstruction makes
    ``gatevision:test-<stem>:gate_pass``; the parser hands us
    ``gate_local="test-<stem>"`` which is NOT in the map. Without this
    helper, ``prosper_gate_uuid`` would fall through to a deterministic
    UUID-v5 hash of the stem — a UUID Prosper has never seen — and the
    upload would 500 on the FK join.

    Strategy: when we detect a test row (either ``gate_local`` starts with
    ``"test-"`` or the DB row's explicit ``source`` column == ``"test"``),
    substitute one of:
      1. ``PROSPER_DEFAULT_TEST_GATE`` env var (operator override)
      2. First key in ``prosper_gate_id_map`` (the canonical configured gate)
      3. The literal string ``"gate-1"`` (last-resort default)

    This way the existing gate_id_map keeps working unchanged AND test rows
    no longer need a separate mapping table.
    """
    is_test_row = (
        (record.get("source") or "").strip().lower() == "test"
        or (gate_local_from_path or "").lower().startswith("test-")
    )
    if not is_test_row:
        return gate_local_from_path
    override = (os.getenv("PROSPER_DEFAULT_TEST_GATE") or "").strip()
    if override:
        return override
    if isinstance(gate_id_map, dict) and gate_id_map:
        first_key = next(iter(gate_id_map.keys()), None)
        if first_key:
            return str(first_key)
    return "gate-1"


def record_to_gate_event_body(
    record: Dict[str, Any],
    *,
    device_id_raw: str,
    gate_id_map: Optional[Dict[str, str]] = None,
    source_system: str = "GateVision",
    image_metadata_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Map a SQLite row to CreateGateEventRequest, or None if not uploadable."""
    gate_local_raw, stage = parse_gatevision_video_path(record.get("video_path"))
    if not gate_local_raw or not stage:
        return None
    event_type = stage_to_gate_event_type(stage)
    if event_type is None:
        return None

    # Test rows need a special lookup so the configured gate UUID applies
    # (see _resolve_gate_local_for_upload for the full rationale).
    gate_local = _resolve_gate_local_for_upload(record, gate_local_raw, gate_id_map)

    device_uuid = prosper_device_uuid(device_id_raw)
    rid = record.get("id", 0)
    ts_raw = record.get("timestamp") or record.get("created_on")
    event_id = prosper_event_uuid(rid, device_uuid, str(ts_raw) if ts_raw is not None else None)

    trailer_raw = record.get("licence_plate_trailer")
    trailer_part, scac_src = _extract_trailer_and_scac(trailer_raw)
    trailer_num = _clean_trailer_number(trailer_part) or _clean_trailer_number(trailer_raw)

    body: Dict[str, Any] = {
        "sourceSystem": source_system,
        "correlationId": str(uuid.uuid4()),
        "eventId": event_id,
        "deviceId": device_uuid,
        "gateId": prosper_gate_uuid(gate_local, gate_id_map),
        "eventType": event_type,
        "sourceTimestamp": _iso_timestamp(ts_raw),
        "damageFlag": False,
        "ocrConfidence": float(record.get("confidence") or 0.0),
    }

    if image_metadata_id:
        body["imageMetadataId"] = image_metadata_id

    # Only send a real trailer number — skip UNKNOWN/empty so Prosper
    # doesn't get polluted with company-name-only OCR failures.
    trailer_clean = trailer_num if (trailer_num and trailer_num != "UNKNOWN") else None
    if trailer_clean:
        body["trailerNumber"] = trailer_clean
    scac_clean = (scac_src or "").strip().upper()[:16] if scac_src else None
    if scac_clean:
        body["scac"] = scac_clean

    cb = os.getenv("PROSPER_CREATED_BY")
    ub = os.getenv("PROSPER_UPDATED_BY")
    if cb:
        body["createdBy"] = cb
    if ub:
        body["updatedBy"] = ub

    return body


def upload_gate_event(
    base_url: str,
    site_id: str,
    body: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
    bearer_token: Optional[str] = None,
    auth_provider: Optional[Any] = None,  # app.prosper_auth.ProsperAuth
    timeout: float = 30.0,
) -> requests.Response:
    """POST one fused gate-event record. Same auth-retry semantics as
    ``upload_trailer_location``: an ``auth_provider`` enables one automatic
    retry with a freshly-logged-in token on 401/403.
    """
    url = f"{base_url.rstrip('/')}/api/sites/{site_id.strip()}/gate-events"
    return _post_with_auth_retry(
        url=url,
        body=body,
        api_key=api_key,
        bearer_token=bearer_token,
        auth_provider=auth_provider,
        timeout=timeout,
        log_prefix="Prosper GateVision",
    )


def upload_gatevision_records_prosper(
    records: List[Dict[str, Any]],
    *,
    base_url: str,
    site_id: str,
    device_id_raw: str,
    gate_id_map: Optional[Dict[str, str]] = None,
    api_key: Optional[str] = None,
    bearer_token: Optional[str] = None,
    auth_provider: Optional[Any] = None,  # app.prosper_auth.ProsperAuth
    source_system: str = "GateVision",
    camera_id_map: Optional[Dict[str, str]] = None,
) -> Tuple[List[int], List[str]]:
    """POST each GateVision fused row to Prosper gate-events. Returns (succeeded_sqlite_ids, errors)."""
    succeeded: List[int] = []
    errors: List[str] = []
    for r in records:
        # Step 1: Upload crop image directly as base64 first to get metadata ID
        image_metadata_id = None
        local_img = (r.get("image_path") or "").strip()
        if local_img and os.path.exists(local_img):
            try:
                from app.prosper_image_upload import upload_image_base64
                trailer_raw = r.get("licence_plate_trailer")
                trailer_part, scac_src = _extract_trailer_and_scac(trailer_raw)
                trailer = _clean_trailer_number(trailer_part) or _clean_trailer_number(trailer_raw)

                log.info(f"[Prosper GateVision] Uploading base64 image for trailer {trailer}...")
                img_id, _ = upload_image_base64(
                    base_url=base_url,
                    site_id=site_id,
                    local_image_path=local_img,
                    trailer_number=trailer,
                    scac=scac_src,
                    device_id_raw=device_id_raw,
                    camera_id=r.get("camera_id"),
                    camera_id_map=camera_id_map,
                    api_key=api_key,
                    bearer_token=bearer_token,
                    auth_provider=auth_provider,
                )
                if img_id:
                    image_metadata_id = img_id
                    log.info(f"[Prosper GateVision] Base64 image uploaded. ID: {img_id}")
            except Exception as ex:
                log.warning(f"[Prosper GateVision] Failed to upload base64 image: {ex}")

        # Step 2: Build gate event body and link the uploaded image
        body = record_to_gate_event_body(
            r,
            device_id_raw=device_id_raw,
            gate_id_map=gate_id_map,
            source_system=source_system,
            image_metadata_id=image_metadata_id,
        )
        if not body:
            errors.append(f"id={r.get('id')}: skip (invalid gatevision path/stage)")
            continue
        try:
            resp = upload_gate_event(
                base_url,
                site_id,
                body,
                api_key=api_key,
                bearer_token=bearer_token,
                auth_provider=auth_provider,
            )
            if resp.ok or resp.status_code in (400, 409):
                rid = r.get("id")
                if rid is not None:
                    succeeded.append(int(rid))
                if not resp.ok:
                    log.info(
                        "[Prosper GateVision] Permanent upload failure for id=%s (HTTP %s): %s. "
                        "Marking as succeeded/deleted to clean queue.",
                        r.get("id"),
                        resp.status_code,
                        (resp.text or "")[:200],
                    )
            else:
                errors.append(f"id={r.get('id')}: HTTP {resp.status_code} {(resp.text or '')[:200]}")
        except requests.RequestException as e:
            errors.append(f"id={r.get('id')}: {e}")
    return succeeded, errors
