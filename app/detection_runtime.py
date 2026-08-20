"""
Live camera detection mode (car vs trailer) — shared by YardVision and GateVision.

Mirrors the model selection rules used by VideoProcessor.set_detection_mode so the
dashboard and APIs stay consistent.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from app.app_logger import get_logger

log = get_logger(__name__)


def build_live_detector(mode: str, globals_cfg: Optional[Dict]) -> Any:
    """
    Build a detector for the given live mode (gate vs stacker).

    Returns None if no detector could be constructed.
    """
    globals_cfg = globals_cfg or {}
    mode = (mode or "gate").strip().lower()
    if mode not in ("gate", "stacker"):
        mode = "gate"

    from app.ai.factory import get_detector
    conf = float(globals_cfg.get("detector_conf", 0.25))

    try:
        detector = get_detector(mode=mode, conf_threshold=conf)
        if detector:
            log.info(f"[detection_runtime] Loaded detector via AI factory for mode: {mode}")
            return detector
    except Exception as e:
        log.warning(f"[detection_runtime] Detector build failed for mode {mode}: {e}")

    return None


def expected_detection_class(detector: Any) -> int:
    """
    Detection dict 'cls' value to keep for tracking (TRT trailer engines use 0).
    """
    if detector is None:
        return 0
    tc = getattr(detector, "target_class", None)
    if tc is not None:
        return int(tc)
    return 0
