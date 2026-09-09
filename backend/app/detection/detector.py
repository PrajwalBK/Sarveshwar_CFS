from typing import Protocol
from app.domain import Detection, Frame


class Detector(Protocol):
    def detect(self, frame: Frame) -> list[Detection]: ...
