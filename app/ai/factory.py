"""
Gate Vision AI Detector Factory Module

Loads the custom trained YOLOv8 Gate Vision detector model for container detection,
container object isolation, and chassis feet extraction.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Any

log = logging.getLogger(__name__)

# Check both possible locations of the trained 3-class container model
MODEL_PATHS = [
    Path("runs/detect/gate_detector/weights/best.pt"),
    Path("runs/detect/stacker_detector/weights/best.pt"),
]


def get_gate_model_path() -> Optional[Path]:
    """Find the active trained Gate YOLO model weights file."""
    for p in MODEL_PATHS:
        if p.exists():
            return p
    return None


def get_detector(mode: str = "gate", conf_threshold: float = 0.25, fallback_detector: Optional[Any] = None) -> Any:
    """
    Factory function to retrieve or initialize the Gate Vision AI detector.

    Args:
        mode: Detection mode (defaults to 'gate')
        conf_threshold: Confidence threshold for bounding boxes
        fallback_detector: Optional pre-initialized detector instance for fallback

    Returns:
        YOLOv8Detector instance
    """
    model_path = get_gate_model_path()

    if model_path is not None:
        log.info(f"[DetectorFactory] ✅ Loading Gate Vision model: {model_path}")
        try:
            from app.ai.detector_yolov8 import YOLOv8Detector
            return YOLOv8Detector(
                model_name=str(model_path),
                conf_threshold=conf_threshold,
                target_class=None,  # Detect all 3 classes: 0='Cointainer', 1='Feet', 2='container_object'
            )
        except Exception as e:
            log.warning(f"[DetectorFactory] Custom Gate model load failed ({e}), using fallback")
            if fallback_detector is not None:
                return fallback_detector

    if fallback_detector is not None:
        return fallback_detector

    try:
        from app.ai.detector_yolov8 import YOLOv8Detector
        return YOLOv8Detector(conf_threshold=conf_threshold, target_class=None)
    except Exception as e:
        log.error(f"[DetectorFactory] Failed to initialize default detector: {e}")
        return None
