from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class VisionPipeline(ABC):
    """Base interface for pluggable vision pipelines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable pipeline name."""

    @abstractmethod
    def process_frame(self, camera_id: str, frame: np.ndarray, frame_count: int) -> None:
        """Process one frame from a specific camera."""

    def flush(self) -> None:
        """Optional periodic flush hook."""
        return None

    def stop(self) -> None:
        """Optional cleanup hook."""
        return None

