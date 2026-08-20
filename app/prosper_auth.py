"""
Prosper Smart Yard API — authentication & token management.

Replaces the pre-existing "set a hand-copied PROSPER_BEARER_TOKEN env var that
expires daily" UX with an automatic login + token cache + refresh-on-401 flow.

Endpoint (per Postman collection):
  POST {base}/api/auth/login
  Content-Type: application/json
  Body: { "siteCode": "...", "email": "...", "password": "..." }
  → returns a JWT bearer token. The collection shows the success status only
  (no body sample), so this module is tolerant of several common JWT-response
  shapes: ``token``, ``accessToken``, ``jwt``, ``bearerToken``, or a plain
  JWT string in the body.

Usage:
    auth = ProsperAuth.from_env(base_url=..., globals_cfg=...)
    if not auth.is_configured():
        # missing credentials → caller should skip uploads gracefully
        ...
    headers = {"Authorization": f"Bearer {auth.get_token()}"}
    resp = requests.post(url, json=body, headers=headers)
    if resp.status_code in (401, 403):
        # token may have expired; force-refresh and retry once
        auth.invalidate()
        headers["Authorization"] = f"Bearer {auth.get_token()}"
        resp = requests.post(url, json=body, headers=headers)

Thread-safe: multiple upload threads can call ``get_token()`` / ``invalidate()``
concurrently; only one ``/api/auth/login`` round-trip runs at a time.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Optional

import requests

from app.app_logger import get_logger

log = get_logger(__name__)


# Status codes that mean "your token is invalid / expired; re-authenticate".
# 401 is the canonical one; 403 is sometimes returned by API gateways when a
# JWT's claims (e.g. role/site) no longer match — also worth a re-login.
AUTH_FAILURE_STATUSES = (401, 403)


class ProsperAuth:
    """Cached-token, auto-refreshing bearer-token provider for Prosper API."""

    def __init__(
        self,
        *,
        base_url: str,
        site_code: Optional[str],
        email: Optional[str],
        password: Optional[str],
        initial_token: Optional[str] = None,
        login_timeout: float = 30.0,
        # Minimum seconds between two successive login attempts. Prevents a
        # broken-credential loop from hammering the API.
        login_cooldown_seconds: float = 5.0,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.site_code = (site_code or "").strip() or None
        self.email = (email or "").strip() or None
        self.password = password or None
        self.login_timeout = float(login_timeout)
        self.login_cooldown_seconds = float(login_cooldown_seconds)

        self._token: Optional[str] = (initial_token or "").strip() or None
        self._lock = threading.Lock()
        self._last_login_attempt_at: float = 0.0
        self._last_login_error: Optional[str] = None

    @classmethod
    def from_env(cls, *, base_url: str, globals_cfg: Optional[Dict[str, Any]] = None) -> "ProsperAuth":
        """Build a ProsperAuth from env vars (preferred) with optional config fallbacks.

        Env vars (read at construction time):
          PROSPER_SITE_CODE   — Prosper site code, e.g. "CHANDLER"
          PROSPER_EMAIL       — login email
          PROSPER_PASSWORD    — login password
          PROSPER_BEARER_TOKEN (optional) — if set, used as the initial cached
                                token so the first upload doesn't have to log
                                in. Once it expires, auto-refresh takes over.

        ``globals_cfg`` keys (config/cameras.yaml ``globals``) checked as fallback:
          prosper_site_code, prosper_email
        Passwords are never read from config files for safety; set
        ``PROSPER_PASSWORD`` in the environment.
        """
        gc = globals_cfg or {}
        site_code = os.getenv("PROSPER_SITE_CODE") or gc.get("prosper_site_code")
        email = os.getenv("PROSPER_EMAIL") or gc.get("prosper_email")
        password = os.getenv("PROSPER_PASSWORD")
        initial = os.getenv("PROSPER_BEARER_TOKEN")
        return cls(
            base_url=base_url,
            site_code=site_code,
            email=email,
            password=password,
            initial_token=initial,
        )

    def is_configured(self) -> bool:
        """True when we can either reuse an initial token or perform a fresh login."""
        if self._token:
            return True
        return bool(self.site_code and self.email and self.password)

    def can_refresh(self) -> bool:
        """True when credentials are sufficient to perform a fresh login."""
        return bool(self.base_url and self.site_code and self.email and self.password)

    def last_error(self) -> Optional[str]:
        """Most recent login error message, or None if last login succeeded / not attempted."""
        with self._lock:
            return self._last_login_error

    def get_token(self) -> Optional[str]:
        """Return a usable bearer token, performing a login round-trip if needed.

        Returns None if neither a cached token nor login credentials are available.
        """
        with self._lock:
            if self._token:
                return self._token
            tok = self._login_locked()
            return tok

    def invalidate(self) -> None:
        """Drop the cached token; the next ``get_token()`` call will re-login."""
        with self._lock:
            if self._token is not None:
                log.info(
                    "[ProsperAuth] Invalidating cached bearer token (will re-login on next request)"
                )
            self._token = None

    def force_refresh(self) -> Optional[str]:
        """Drop cache + immediately log in. Returns the new token or None on failure."""
        with self._lock:
            self._token = None
            return self._login_locked()

    # ---------------------------------------------------------------- internal

    def _login_locked(self) -> Optional[str]:
        """Run /api/auth/login under self._lock. Returns new token or None."""
        if not self.can_refresh():
            self._last_login_error = (
                "Cannot refresh: PROSPER_SITE_CODE / PROSPER_EMAIL / PROSPER_PASSWORD not all set "
                "(and no initial PROSPER_BEARER_TOKEN). Set them in the environment so the edge "
                "can self-authenticate against Prosper."
            )
            log.warning("[ProsperAuth] %s", self._last_login_error)
            return None

        # Cooldown — don't hammer the login endpoint after a bad password.
        now = time.monotonic()
        wait_left = self.login_cooldown_seconds - (now - self._last_login_attempt_at)
        if wait_left > 0:
            log.info(
                "[ProsperAuth] login cooldown active (%.1fs remaining); skipping until next call",
                wait_left,
            )
            return None
        self._last_login_attempt_at = now

        url = f"{self.base_url}/api/auth/login"
        body = {
            "siteCode": self.site_code,
            "email": self.email,
            "password": self.password,
        }
        log.info("[ProsperAuth] POST %s siteCode=%s email=%s", url, self.site_code, self.email)
        try:
            resp = requests.post(
                url,
                json=body,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=self.login_timeout,
            )
        except requests.RequestException as e:
            self._last_login_error = f"login network error: {e}"
            log.warning("[ProsperAuth] %s", self._last_login_error)
            return None

        if not resp.ok:
            snippet = (resp.text or "")[:200].replace("\n", " ")
            self._last_login_error = f"login HTTP {resp.status_code}: {snippet}"
            log.warning("[ProsperAuth] %s", self._last_login_error)
            return None

        token = _extract_token_from_response(resp)
        if not token:
            snippet = (resp.text or "")[:200].replace("\n", " ")
            self._last_login_error = (
                f"login returned HTTP {resp.status_code} but no JWT token in response: {snippet}"
            )
            log.warning("[ProsperAuth] %s", self._last_login_error)
            return None

        self._token = token
        self._last_login_error = None
        preview = token[:12] + "..." + token[-4:] if len(token) > 24 else "<short>"
        log.info("[ProsperAuth] Login succeeded; cached new bearer token (%s)", preview)
        return token


def _extract_token_from_response(resp: requests.Response) -> Optional[str]:
    """Pull a JWT out of an auth/login response, tolerant of several shapes."""
    # JSON object with one of the common keys
    try:
        data = resp.json()
    except ValueError:
        data = None

    if isinstance(data, dict):
        # Common shapes seen across Prosper-like APIs.
        for key in ("token", "accessToken", "access_token", "jwt", "bearerToken", "bearer_token"):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # Some APIs nest the token under a ``data`` or ``result`` envelope.
        for envelope in ("data", "result"):
            inner = data.get(envelope)
            if isinstance(inner, dict):
                for key in ("token", "accessToken", "access_token", "jwt"):
                    v = inner.get(key)
                    if isinstance(v, str) and v.strip():
                        return v.strip()

    if isinstance(data, str) and data.count(".") >= 2:
        return data.strip()

    # Plain-text response that looks like a JWT (three dot-separated parts).
    txt = (resp.text or "").strip().strip('"')
    if txt.count(".") >= 2 and " " not in txt and len(txt) < 4096:
        return txt
    return None
