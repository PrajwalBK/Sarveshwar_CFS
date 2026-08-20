"""
GateVision / YardVision → Prosper Smart Yard API image metadata upload.

Endpoint (per Postman collection):
  POST {base}/api/sites/{siteId}/images/trailers
  body: {
    sourceSystem, correlationId, entityType, entityId,
    fileName, contentType, fileUrl, thumbnailUrl,
    capturedUtc, deviceId, cameraId, metadataJson, createdBy, updatedBy
  }

This endpoint is *metadata-only*: ``fileUrl`` is a publicly-reachable URL
where Prosper can pull the actual JPEG from. The edge serves the image bytes
through its own Flask metrics server at ``/api/edge-images/<rel-path>`` and
this module just sends Prosper a pointer to that URL.

Set ``EDGE_PUBLIC_BASE_URL`` (e.g. ``http://192.168.1.89:8080`` or
``https://edge.example.com``) so the constructed ``fileUrl`` is reachable
from wherever the Prosper backend lives. When the env var is unset image
upload is skipped entirely — the gate/yard event upload still works.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

from app.app_logger import get_logger
from app.container_utils import (
    _is_uuid,
    _iso_timestamp,
    _post_with_auth_retry,
    prosper_device_uuid,
    prosper_event_uuid,
)

log = get_logger(__name__)


# Image entityType is undocumented in the Postman collection — the endpoint
# itself lives under /images/trailers, so we default to "Trailer". Operators
# can override via env PROSPER_IMAGE_ENTITY_TYPE if their backend uses a
# different vocabulary (e.g. "GateEvent" / "TrailerLocation").
DEFAULT_ENTITY_TYPE = "Trailer"


# Subdirectories of ``out/`` that we're willing to expose to Prosper as
# fetchable image URLs. ``crops`` is where the test-mode/yard pipeline saves
# JPEGs; ``gatevision_evidence`` is where the LIVE GateVisionPipeline saves
# per-vehicle gate-pass JPEGs (see GateVisionPipeline._save_evidence_image).
# Anything not under one of these roots is refused — we never want to serve
# arbitrary filesystem paths.
EXPOSED_IMAGE_ROOTS = ("crops", "gatevision_evidence")


def _strip_to_known_image_root(image_path: str) -> Optional[str]:
    """Return the path relative to ``out/`` if it falls under one of the
    allowlisted image roots. The returned string keeps the root segment
    (e.g. ``"crops/<cam>/<stem>/file.jpg"`` or
    ``"gatevision_evidence/<gate>/<cam>/file.jpg"``) so the Flask server's
    /api/edge-images/<path> handler can serve from the same shape.

    Returns None for paths outside ``out/{EXPOSED_IMAGE_ROOTS}/...``.
    """
    if not image_path:
        return None
    p = Path(image_path)
    parts = list(p.parts)
    if "out" not in parts:
        return None
    i = parts.index("out")
    if i + 1 >= len(parts):
        return None
    root = parts[i + 1]
    if root not in EXPOSED_IMAGE_ROOTS:
        return None
    # Keep root segment so the URL/server agree on what to serve.
    rel = "/".join(parts[i + 1 :])
    return rel or None


# Back-compat alias: older callers (and a few tests) imported this name. The
# behavior is now broader (also serves gatevision_evidence) but the contract
# is the same — None when not exposable, otherwise a relative path safe for
# the Flask /api/edge-images route.
_strip_out_crops_prefix = _strip_to_known_image_root


def edge_image_url(image_path: str, edge_public_base_url: str) -> Optional[str]:
    """Build a publicly-reachable URL for an evidence/crop image, or None if
    the local path isn't under an exposed root."""
    rel = _strip_to_known_image_root(image_path)
    if rel is None:
        return None
    base = (edge_public_base_url or "").rstrip("/")
    if not base:
        return None
    # Each path segment is URL-encoded; "/" itself is preserved.
    encoded = "/".join(quote(seg, safe="") for seg in rel.split("/"))
    return f"{base}/api/edge-images/{encoded}"


def prosper_camera_uuid(
    local_camera_id: Optional[str], camera_id_map: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """Map a local camera id (e.g. ``lifecam-hd6000-01``) to a registered Prosper
    camera UUID via ``camera_id_map`` (typically ``globals.prosper_camera_id_map``
    in cameras.yaml). Returns None when no mapping is available — the image
    endpoint then receives a body without ``cameraId``.
    """
    raw = (local_camera_id or "").strip()
    if not raw:
        return None
    if camera_id_map:
        mapped = camera_id_map.get(raw) or camera_id_map.get(raw.lower())
        if mapped and _is_uuid(str(mapped)):
            return str(uuid.UUID(str(mapped).strip())).lower()
    if _is_uuid(raw):
        return str(uuid.UUID(raw)).lower()
    return None


def build_image_body_for_record(
    record: Dict[str, Any],
    *,
    device_id_raw: str,
    edge_public_base_url: str,
    entity_type: str = DEFAULT_ENTITY_TYPE,
    source_system: str = "GateVision",
    camera_id_map: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Construct the POST body for ``/images/trailers`` from a SQLite row.

    Returns None when the record lacks a usable image_path or the path can't
    be turned into a public URL.
    """
    image_path = (record.get("image_path") or "").strip()
    if not image_path:
        return None
    file_url = edge_image_url(image_path, edge_public_base_url)
    if not file_url:
        return None

    file_name = Path(image_path).name
    content_type, _ = mimetypes.guess_type(file_name)
    if not content_type:
        content_type = "image/jpeg"

    device_uuid = prosper_device_uuid(device_id_raw)
    rid = record.get("id", 0)
    ts_raw = record.get("timestamp") or record.get("created_on")
    # entityId is intentionally the SAME deterministic UUID we send as ``eventId``
    # for the corresponding gate-event / yard-location upload, so the image row
    # cross-references that event idempotently. Operators whose backend expects
    # a real Trailer UUID instead can run the lookup themselves and adjust this.
    entity_id = prosper_event_uuid(rid, device_uuid, str(ts_raw) if ts_raw is not None else None)

    cam_uuid = prosper_camera_uuid(record.get("camera_id"), camera_id_map)

    metadata = {
        "edge_sqlite_id": rid,
        "trailerNumber": record.get("licence_plate_trailer"),
        "ocrConfidence": float(record.get("confidence") or 0.0),
        "videoPath": record.get("video_path"),
        "frameNumber": record.get("frame_number"),
        "trackId": record.get("track_id"),
    }

    body: Dict[str, Any] = {
        "sourceSystem": source_system,
        "correlationId": str(uuid.uuid4()),
        "entityType": entity_type,
        "entityId": entity_id,
        "fileName": file_name,
        "contentType": content_type,
        "fileUrl": file_url,
        "capturedUtc": _iso_timestamp(ts_raw),
        "deviceId": device_uuid,
        "metadataJson": json.dumps(metadata, default=str),
    }
    if cam_uuid:
        body["cameraId"] = cam_uuid

    cb = os.getenv("PROSPER_CREATED_BY")
    ub = os.getenv("PROSPER_UPDATED_BY")
    if cb:
        body["createdBy"] = cb
    if ub:
        body["updatedBy"] = ub
    return body


def upload_trailer_image(
    base_url: str,
    site_id: str,
    body: Dict[str, Any],
    *,
    api_key: Optional[str] = None,
    bearer_token: Optional[str] = None,
    auth_provider: Optional[Any] = None,  # app.prosper_auth.ProsperAuth
    timeout: float = 30.0,
) -> requests.Response:
    """POST one image-metadata record; retries once with a fresh token on 401/403."""
    url = f"{base_url.rstrip('/')}/api/sites/{site_id.strip()}/images/trailers"
    return _post_with_auth_retry(
        url=url,
        body=body,
        api_key=api_key,
        bearer_token=bearer_token,
        auth_provider=auth_provider,
        timeout=timeout,
        log_prefix="Prosper Image",
    )


def upload_record_images(
    records: List[Dict[str, Any]],
    *,
    base_url: str,
    site_id: str,
    device_id_raw: str,
    edge_public_base_url: str,
    camera_id_map: Optional[Dict[str, str]] = None,
    api_key: Optional[str] = None,
    bearer_token: Optional[str] = None,
    auth_provider: Optional[Any] = None,
    entity_type: str = DEFAULT_ENTITY_TYPE,
    source_system: str = "GateVision",
) -> Tuple[List[int], List[str]]:
    """POST an image record for each row that has an exposable image_path.

    Returns (succeeded_sqlite_ids, error_messages). Image upload errors do
    NOT roll back the parent gate-event / yard-location row — the caller
    should treat this as best-effort enrichment.
    """
    if not edge_public_base_url:
        return [], []  # silently skip when not configured

    succeeded: List[int] = []
    errors: List[str] = []
    for r in records:
        body = build_image_body_for_record(
            r,
            device_id_raw=device_id_raw,
            edge_public_base_url=edge_public_base_url,
            camera_id_map=camera_id_map,
            entity_type=entity_type,
            source_system=source_system,
        )
        if body is None:
            # Most common reason: image_path missing or not under out/crops/
            continue
        try:
            resp = upload_trailer_image(
                base_url,
                site_id,
                body,
                api_key=api_key,
                bearer_token=bearer_token,
                auth_provider=auth_provider,
            )
            if resp.ok:
                rid = r.get("id")
                if rid is not None:
                    succeeded.append(int(rid))
            else:
                snippet = (resp.text or "")[:200]
                errors.append(f"id={r.get('id')} (image): HTTP {resp.status_code} {snippet}")
        except requests.RequestException as e:
            errors.append(f"id={r.get('id')} (image): {e}")
    return succeeded, errors


def upload_image_base64(
    base_url: str,
    site_id: str,
    local_image_path: str,
    *,
    trailer_number: str,
    scac: Optional[str] = None,
    device_id_raw: str,
    camera_id: Optional[str] = None,
    camera_id_map: Optional[Dict[str, str]] = None,
    api_key: Optional[str] = None,
    bearer_token: Optional[str] = None,
    auth_provider: Optional[Any] = None,
    timeout: float = 30.0,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Step 1: Upload crop image directly as base64 to /api/sites/{siteId}/images/trailers/upload-s3.
    Returns (imageMetadataId, s3_image_path) if successful, otherwise (None, None).
    """
    p = Path(local_image_path)
    if not p.is_file():
        log.warning(f"[Prosper Image Base64] Image file not found: {local_image_path}")
        return None, None

    try:
        with open(p, "rb") as image_file:
            encoded_bytes = base64.b64encode(image_file.read())
            encoded_string = encoded_bytes.decode('utf-8')
    except Exception as e:
        log.error(f"[Prosper Image Base64] Failed to read/encode image {local_image_path}: {e}")
        return None, None

    # Format base64 data URI (defaulting to image/jpeg)
    ext = p.suffix.lower()
    mime_type = "image/jpeg"
    if ext == ".png":
        mime_type = "image/png"
    elif ext in (".jpg", ".jpeg"):
        mime_type = "image/jpeg"
    base64_data_uri = f"data:{mime_type};base64,{encoded_string}"

    device_uuid = prosper_device_uuid(device_id_raw)
    cam_uuid = prosper_camera_uuid(camera_id, camera_id_map)

    scac_clean = (scac or "").strip().upper()
    if not scac_clean or scac_clean == "UNKNOWN":
        scac_clean = "UNKN"

    body = {
        "trailerNumber": trailer_number,
        "scac": scac_clean,
        "deviceId": device_uuid,
        "cameraId": cam_uuid,
        "imagePath": base64_data_uri
    }

    url = f"{base_url.rstrip('/')}/api/sites/{site_id.strip()}/images/trailers/upload-s3"
    try:
        resp = _post_with_auth_retry(
            url=url,
            body=body,
            api_key=api_key,
            bearer_token=bearer_token,
            auth_provider=auth_provider,
            timeout=timeout,
            log_prefix="Prosper Image Base64",
        )
        if resp.ok:
            try:
                data = resp.json()
                log.info(f"[Prosper Image Base64] Response JSON: {data}")
                
                nested = data.get("data") if isinstance(data.get("data"), dict) else {}
                img_id = data.get("id") or nested.get("id") or nested.get("imageMetadataId") or data.get("imageMetadataId")
                img_path = nested.get("fileUrl") or nested.get("imagePath") or data.get("fileUrl") or data.get("imagePath")
                
                return img_id, img_path
            except Exception as json_err:
                log.warning(f"[Prosper Image Base64] Failed to parse successful response JSON: {json_err}")
                return None, None
        else:
            snippet = (resp.text or "")[:200]
            log.warning(f"[Prosper Image Base64] Upload failed: HTTP {resp.status_code} {snippet}")
            return None, None
    except Exception as e:
        log.warning(f"[Prosper Image Base64] Exception during upload request: {e}")
        return None, None
