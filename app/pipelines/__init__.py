from __future__ import annotations

from app.app_logger import get_logger
from app.pipelines.gate_pipeline import GateVisionPipeline

log = get_logger(__name__)


def create_pipeline(app):
    """
    Factory for Vision Pipeline — Dedicated Gate Vision System.
    """
    gate = GateVisionPipeline(
        detector=app.detector,
        trackers=app.trackers,
        preprocessor=app.preprocessor,
        ocr=app.ocr,
        publisher=app.publisher,
        metrics_server=app.metrics_server,
        camera_metrics=app.camera_metrics,
        latest_frames=app.latest_frames,
        frame_lock=app.frame_lock,
        config=app.config,
        ingest_enabled=app.ingest_enabled,
        ingest_url=app.ingest_url,
        db_record_sink=app.store_gatevision_record_in_db,
        detector_runtime_lock=getattr(app, "detector_runtime_lock", None),
    )
    log.info("[create_pipeline] Initialized dedicated GateVisionPipeline")
    return gate
