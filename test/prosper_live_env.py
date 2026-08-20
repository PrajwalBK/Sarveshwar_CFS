"""
Load Prosper credentials from the repo-root `.env` for integration-style tests.

Live POST tests run only when **both** are true:

1. ``RUN_PROSPER_LIVE_TESTS=1`` (or ``true`` / ``yes``) — set in ``.env`` or the shell so normal
   ``pytest`` runs do not call Prosper unless you opt in.
2. ``PROSPER_SITE_ID`` plus ``DASHBOARD_API_KEY`` / ``PROSPER_API_KEY`` / ``PROSPER_BEARER_TOKEN``.

Gate live upload additionally needs ``PROSPER_TEST_GATE_ID`` (UUID of an existing gate in Prosper).

Default ``pytest`` excludes integration tests (see ``pytest.ini``). Run: ``pytest -m integration``.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


def _valid_uuid(s: str) -> bool:
    try:
        uuid.UUID(str(s).strip())
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_dotenv_from_repo() -> None:
    """Load ``.env`` from project root (same pattern as ``app.main_trt_demo``)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    p = repo_root() / ".env"
    if p.is_file():
        load_dotenv(p)


def get_prosper_live_settings() -> Optional[Dict[str, Any]]:
    """
    Returns dict with base_url, site_id, api_key, bearer_token, device_id_raw — or None if not configured.
    Loads ``.env`` from repo root internally. Does **not** check ``RUN_PROSPER_LIVE_TESTS``; tests should
    call ``prosper_live_tests_enabled()`` first for a clear skip reason.
    """
    load_dotenv_from_repo()
    from app.prosper_yard_upload import prosper_site_id_valid

    base_url = (os.getenv("PROSPER_API_BASE_URL") or "http://syapi.prosperassettracking.com").rstrip("/")
    site_id = (os.getenv("PROSPER_SITE_ID") or "").strip()
    api_key = os.getenv("DASHBOARD_API_KEY") or os.getenv("PROSPER_API_KEY")
    bearer = os.getenv("PROSPER_BEARER_TOKEN")
    if not prosper_site_id_valid(site_id):
        return None
    if not (api_key or bearer):
        return None
    device_raw = (
        (os.getenv("PROSPER_DEVICE_ID") or os.getenv("PROSPER_DEVICE_UUID") or "").strip()
        or (os.getenv("EDGE_DEVICE_ID") or "pytest-prosper-live").strip()
    )
    return {
        "base_url": base_url,
        "site_id": site_id,
        "api_key": api_key.strip() if api_key else None,
        "bearer_token": bearer.strip() if bearer else None,
        "device_id_raw": device_raw,
    }


PROSPER_LIVE_OPT_IN_SKIP_REASON = (
    "Set RUN_PROSPER_LIVE_TESTS=1 in .env or environment to enable live Prosper POST tests."
)

PROSPER_LIVE_CREDS_SKIP_REASON = (
    "With RUN_PROSPER_LIVE_TESTS=1, set repo-root .env: PROSPER_SITE_ID (UUID) and "
    "DASHBOARD_API_KEY or PROSPER_API_KEY or PROSPER_BEARER_TOKEN."
)

PROSPER_LIVE_DEVICE_SKIP_REASON = (
    "Live POST tests need a Prosper-registered device: set PROSPER_DEVICE_ID (UUID), or set "
    "EDGE_DEVICE_ID to that same UUID string. A non-UUID EDGE_DEVICE_ID is mapped to a synthetic UUID "
    "that the API often rejects with HTTP 500. Register devices via POST /api/sites/{siteId}/devices."
)


def prosper_live_device_configured() -> bool:
    """True if we will send a literal device UUID Prosper can resolve (not only a uuid5 alias)."""
    load_dotenv_from_repo()
    if (os.getenv("PROSPER_DEVICE_ID") or os.getenv("PROSPER_DEVICE_UUID") or "").strip():
        return True
    edge = (os.getenv("EDGE_DEVICE_ID") or "").strip()
    return bool(edge and _valid_uuid(edge))


def prosper_live_tests_enabled() -> bool:
    load_dotenv_from_repo()
    v = (os.getenv("RUN_PROSPER_LIVE_TESTS") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def get_prosper_test_gate_id_map() -> Optional[Dict[str, str]]:
    """If PROSPER_TEST_GATE_ID is a UUID, return {\"gate-1\": id} for live gate-event tests."""
    load_dotenv_from_repo()
    if not prosper_live_tests_enabled():
        return None
    gid = (os.getenv("PROSPER_TEST_GATE_ID") or "").strip()
    if not gid or not _valid_uuid(gid):
        return None
    return {"gate-1": str(uuid.UUID(gid)).lower()}


PROSPER_GATE_LIVE_SKIP_REASON = (
    "Live gate-event test needs PROSPER_TEST_GATE_ID (UUID of a gate that exists in Prosper for this site)."
)
