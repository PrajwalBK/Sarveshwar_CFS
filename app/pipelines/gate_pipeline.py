from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import threading
import time
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
import requests

from app.app_logger import get_logger
from app.detection_runtime import expected_detection_class
from app.gatevision.fusion import CameraObservation, GateFusionEngine
from app.pipelines.base_pipeline import VisionPipeline

log = get_logger(__name__)


@dataclass
class _PendingGateObservation:
    """An observation captured by the live thread, awaiting offline OCR.

    Mirrors CameraObservation's pre-OCR fields so we can construct the final
    CameraObservation in the offline worker once `text`/`conf` are filled.
    """

    camera_id: str
    camera_role: str
    gate_id: str
    ts: datetime
    track_id: int
    bbox: List[float]
    crop: np.ndarray
    evidence_image_path: Optional[str]
    frame_count: int


class GateVisionPipeline(VisionPipeline):
    """
    GateVision: per gate camera, detect trailer or car (shared live detector mode),
    track, crop each track's bounding box from the frame, then run OCR on that crop.
    Observations feed the fusion engine for multi-camera gate events.
    """

    def __init__(
        self,
        detector,
        trackers: Dict[str, object],
        preprocessor,
        ocr,
        publisher,
        metrics_server,
        camera_metrics: Dict[str, dict],
        latest_frames: Dict[str, np.ndarray],
        frame_lock,
        config: Dict,
        ingest_enabled: bool = False,
        ingest_url: str = "",
        db_record_sink: Optional[Callable[[dict], None]] = None,
        detector_runtime_lock=None,
    ):
        self.detector = detector
        self.trackers = trackers
        self.preprocessor = preprocessor
        self.ocr = ocr
        self.publisher = publisher
        self.metrics_server = metrics_server
        self.camera_metrics = camera_metrics
        self.latest_frames = latest_frames
        self.frame_lock = frame_lock
        self.config = config or {}
        self.ingest_enabled = ingest_enabled
        self.ingest_url = ingest_url
        self.db_record_sink = db_record_sink
        self._detector_runtime_lock = detector_runtime_lock

        globals_cfg = self.config.get("globals", {})
        gate_cfg = self.config.get("gatevision", {})
        self.detect_every_n = int(globals_cfg.get("detect_every_n", 5))
        self.ocr_run_every_n_frames = int(gate_cfg.get("ocr_run_every_n_frames", 6))
        self.ocr_cache: Dict[tuple, dict] = {}
        self.ocr_cache_max_age = int(gate_cfg.get("ocr_cache_max_age", 25))
        self.ocr_min_conf = float(gate_cfg.get("ocr_min_confidence", 0.5))

        self.camera_roles = {}
        self.camera_gate_ids = {}
        for c in self.config.get("cameras", []):
            cid = c.get("id")
            if not cid:
                continue
            self.camera_roles[cid] = c.get("role", "unknown")
            self.camera_gate_ids[cid] = c.get("gate_id", "gate-1")

        self.fusion = GateFusionEngine(
            fusion_window_seconds=float(gate_cfg.get("fusion_window_seconds", 3.0)),
            finalize_after_seconds=float(gate_cfg.get("finalize_after_seconds", 1.5)),
            min_roles_required=int(gate_cfg.get("min_roles_required", 1)),
        )
        self._recent_gate_events = []
        self._max_recent_events = int(gate_cfg.get("max_recent_events", 50))
        self._stats = {
            "events_emitted": 0,
            "confirmed_events": 0,
            "needs_review_events": 0,
            "last_event_at": None,
        }
        self._event_review_overrides: Dict[str, dict] = {}
        self._evidence_dir = Path("out/gatevision_evidence")
        self._evidence_dir.mkdir(parents=True, exist_ok=True)
        self._apply_gate_config(gate_cfg)
        # When set to ProcessingQueueManager.gpu_lock, live YOLO yields while offline
        # video+OCR jobs run (avoids Jetson OOM from overlapping CUDA workloads).
        # The offline gate-OCR worker below also acquires this lock so that GateVision's
        # OCR is serialized with YardVision's offline OCR/video processing — i.e. YOLO
        # and Qwen3-VL never run CUDA kernels concurrently.
        self.offline_gpu_lock: Optional[threading.Lock] = None

        # ---- Offline OCR (YardVision-style two-stage processing) -----------------
        # Live thread: detect + track + crop + buffer here.
        # Offline thread: drain → OCR → fuse → emit events.
        self._pending_obs: List[_PendingGateObservation] = []
        self._pending_obs_lock = threading.Lock()
        self._pending_obs_max = int(gate_cfg.get("offline_ocr_pending_max", 500))
        self._offline_poll_interval = float(gate_cfg.get("offline_ocr_poll_interval_seconds", 0.5))
        self._offline_running = True
        self._offline_thread = threading.Thread(
            target=self._offline_ocr_worker,
            name="GateVision-offline-ocr",
            daemon=True,
        )
        self._offline_thread.start()
        log.info(
            "[GateVisionPipeline] Offline OCR worker started (poll=%.2fs, pending_max=%d) — "
            "OCR runs out-of-band like YardVision so live YOLO and Qwen3-VL don't compete on GPU.",
            self._offline_poll_interval,
            self._pending_obs_max,
        )

    @property
    def name(self) -> str:
        return "gatevision"

    def process_frame(self, camera_id: str, frame: np.ndarray, frame_count: int) -> None:
        """
        Detect target class (trailer or car) → ByteTrack → crop bbox from frame → OCR on crop.
        Only tracks matching the active detector class are OCR'd (same rule as YardVision).
        """
        # Memory-saver: while an offline gate-test (or yard-test) job holds
        # ``offline_gpu_lock``, drop the live frame entirely. The frame.copy()
        # into latest_frames, preprocessing, tracker update, crop buffering
        # and event emission below all compete with the offline worker for
        # the same Tegra unified-memory pool on Jetson, and that pressure has
        # been the proximate cause of OOM during gate-test OCR. The dashboard
        # /stream endpoint will keep showing the previously cached frame
        # (frozen preview) for the duration of the test, then resume
        # automatically when the offline worker releases the lock.
        olock = self.offline_gpu_lock
        if olock is not None and olock.locked():
            return

        with self.frame_lock:
            self.latest_frames[camera_id] = frame.copy()

        if camera_id not in self.trackers:
            from app.ai.tracker_bytetrack import ByteTrackWrapper
            self.trackers[camera_id] = ByteTrackWrapper(track_thresh=0.20)
        if camera_id not in self.camera_metrics:
            self.camera_metrics[camera_id] = {
                'frames_processed': 0,
                'fps_ema': 0.0,
                'last_publish': None
            }

        metrics = self.camera_metrics[camera_id]
        tracker = self.trackers[camera_id]

        processed = frame
        if self.preprocessor and self.preprocessor.enable_yolo_preprocessing:
            processed = self.preprocessor.preprocess_for_yolo(frame)

        want_cls = expected_detection_class(self.detector) if self.detector else -1

        detections = []
        if self.detector and frame_count % self.detect_every_n == 0:
            olock = self.offline_gpu_lock
            all_detections = []
            if olock is not None:
                if olock.acquire(blocking=False):
                    try:
                        lock = self._detector_runtime_lock
                        ctx = lock if lock is not None else nullcontext()
                        with ctx:
                            all_detections = self.detector.detect(processed)
                    finally:
                        olock.release()
            else:
                lock = self._detector_runtime_lock
                ctx = lock if lock is not None else nullcontext()
                with ctx:
                    all_detections = self.detector.detect(processed)
            for det in all_detections:
                if det.get("cls", -1) == want_cls:
                    detections.append(det)

        tracks = tracker.update(detections, frame)
        now = datetime.utcnow()
        camera_role = self.camera_roles.get(camera_id, "unknown")
        gate_id = self.camera_gate_ids.get(camera_id, "gate-1")

        for trk in tracks:
            track_id = trk.get("track_id")
            track_cls = trk.get("cls", -1)
            if track_cls != -1 and want_cls != -1 and track_cls != want_cls:
                continue

            x1, y1, x2, y2 = [int(v) for v in trk.get("bbox", [0, 0, 0, 0])]
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Skip OCR on tiny crops (likely false positives), aligned with YardVision heuristic
            crop_area = (x2 - x1) * (y2 - y1)
            min_crop_area = 1000
            if crop_area < min_crop_area:
                continue

            # Save evidence image immediately (cheap CPU op, no GPU). Then hand the crop
            # off to the offline OCR worker — fusion / event emission happens there, not here.
            evidence_path = self._save_evidence_image(camera_id, gate_id, int(track_id), now, crop)
            self._enqueue_pending_observation(
                _PendingGateObservation(
                    camera_id=camera_id,
                    camera_role=camera_role,
                    gate_id=gate_id,
                    ts=now,
                    track_id=int(track_id),
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    crop=crop.copy(),  # frame buffer is reused on next iteration
                    evidence_image_path=evidence_path,
                    frame_count=frame_count,
                )
            )

        metrics["frames_processed"] += 1
        if metrics["frames_processed"] == 1:
            metrics["fps_ema"] = 30.0
        else:
            metrics["fps_ema"] = 0.1 * 30.0 + 0.9 * metrics["fps_ema"]
        metrics["last_publish"] = now

        queue_depth = self.publisher.get_queue_depth()
        self.metrics_server.update_camera_metrics(
            camera_id,
            metrics["fps_ema"],
            metrics["frames_processed"],
            metrics["last_publish"],
            queue_depth,
        )

    # ---------- Offline OCR worker (YardVision-style two-stage processing) ----------

    def _enqueue_pending_observation(self, pending: _PendingGateObservation) -> None:
        """Buffer an observation captured by the live thread for later OCR + fusion.

        Caps the buffer to avoid unbounded growth if OCR can't keep up (e.g. OCR
        not yet loaded, or offline_gpu_lock held for a long time by YardVision
        video processing). When the cap is hit, the oldest observations are
        dropped — fusion would have aged them out anyway.
        """
        with self._pending_obs_lock:
            self._pending_obs.append(pending)
            overflow = len(self._pending_obs) - self._pending_obs_max
            if overflow > 0:
                dropped = self._pending_obs[:overflow]
                self._pending_obs = self._pending_obs[overflow:]
                
                # Rate limit the warning log to once every 5 seconds to avoid log flooding
                now_s = time.time()
                last_warn = getattr(self, "_last_backlog_warn_time", 0.0)
                if now_s - last_warn > 5.0:
                    self._last_backlog_warn_time = now_s
                    log.warning(
                        "[GateVision] Offline OCR backlog full; dropped %d oldest observation(s). "
                        "Consider raising gatevision.offline_ocr_pending_max or speeding up OCR. "
                        "(This warning is rate-limited to once every 5 seconds)",
                        len(dropped),
                    )

    def _drain_pending_observations(self) -> List[_PendingGateObservation]:
        with self._pending_obs_lock:
            if not self._pending_obs:
                return []
            batch = self._pending_obs
            self._pending_obs = []
            return batch

    def _offline_ocr_worker(self) -> None:
        """Background worker: drain pending obs → OCR (under GPU lock) → fuse → emit.

        Mirrors YardVision's offline batch-OCR pattern (see app/processing_queue.py).
        While this thread holds `offline_gpu_lock`, the live pipeline's YOLO inference
        yields (see process_frame), so YOLO and Qwen3-VL never run kernels at the
        same time — the same arrangement that lets YardVision run Qwen3-VL-4B on
        Jetson without OOM.
        """
    def _offline_ocr_worker(self) -> None:
        """Background worker: drain pending obs → OCR (under GPU lock) → fuse → emit."""
        while self._offline_running and not getattr(self, '_stop_event', threading.Event()).is_set():
            try:
                # Light pacing using interruptible stop event wait
                stop_evt = getattr(self, '_stop_event', None)
                if stop_evt and stop_evt.wait(timeout=self._offline_poll_interval):
                    break
                elif not stop_evt:
                    time.sleep(self._offline_poll_interval)

                if self.ocr is None:
                    # OCR not loaded yet (lazy init in main_trt_demo); keep buffering.
                    # We still call collect_ready_events() so candidates land.
                    with self._pending_obs_lock:
                        p_len = len(self._pending_obs)
                    if p_len > 0:
                        now_s = time.time()
                        last_no_ocr_warn = getattr(self, "_last_no_ocr_warn_time", 0.0)
                        if now_s - last_no_ocr_warn > 10.0:
                            self._last_no_ocr_warn_time = now_s
                            log.warning(
                                "[GateVision] OCR engine is not initialized yet (self.ocr is None). "
                                "%d crop observation(s) buffered waiting for OCR to load.",
                                p_len,
                            )
                    self._collect_and_emit_events()
                    continue

                batch = self._drain_pending_observations()
                if not batch:
                    self._collect_and_emit_events()
                    continue

                self._batch_ocr_and_fuse(batch)
                self._collect_and_emit_events()
            except Exception as exc:
                log.exception("[GateVision] Offline OCR worker iteration failed: %s", exc)
                # Avoid tight error loop
                time.sleep(0.5)

    def _batch_ocr_and_fuse(self, batch: List[_PendingGateObservation]) -> None:
        """OCR every pending observation under the GPU lock, then push to fusion.

        OCR results are also written to the DB sink with stage='candidate', matching
        the previous live-OCR behavior (one candidate row per observation).
        """
        olock = self.offline_gpu_lock
        # Block here, like YardVision's OCR worker — we *want* to wait if YardVision
        # is currently doing video processing or its own OCR.
        ctx = olock if olock is not None else nullcontext()
        with ctx:
            for pending in batch:
                text, conf = self._ocr_pending(pending)
                # Update the per-track cache so /api/gatevision/status reflects the
                # latest read (cache key kept for backward compat with consumers).
                self.ocr_cache[(pending.camera_id, pending.track_id)] = {
                    "text": text,
                    "conf": conf,
                    "last_updated": pending.frame_count,
                }
                obs = CameraObservation(
                    camera_id=pending.camera_id,
                    camera_role=pending.camera_role,
                    gate_id=pending.gate_id,
                    ts=pending.ts,
                    track_id=pending.track_id,
                    text=text,
                    conf=float(conf),
                    bbox=pending.bbox,
                    evidence_image_path=pending.evidence_image_path,
                )
                self.fusion.add_observation(obs)
                self._store_detection_record(
                    gate_id=pending.gate_id,
                    camera_id=pending.camera_id,
                    track_id=pending.track_id,
                    frame_count=pending.frame_count,
                    ts=pending.ts,
                    text=text,
                    conf=float(conf),
                    evidence_image_path=pending.evidence_image_path,
                    stage="candidate",
                )

    def _ocr_pending(self, pending: _PendingGateObservation) -> tuple:
        """Run OCR on a buffered crop. Caller must hold offline_gpu_lock."""
        if self.ocr is None:
            return "", 0.0

        local_crop = pending.crop
        h_crop, w_crop = local_crop.shape[:2]

        # Bring the crop into the 640–1024 px range for the VLM.
        # • If it's too large (>1024 px), shrink so we don't OOM.
        # • If it's too small (<640 px), upscale with INTER_CUBIC so painted
        #   digits are sharp enough for Qwen-VL to distinguish.
        #   Trailer crops are often only 150–350 px wide from far-away cameras.
        TARGET_MIN = 640
        TARGET_MAX = 1024
        max_dim = max(h_crop, w_crop)
        if max_dim > TARGET_MAX:
            scale = TARGET_MAX / max_dim
            local_crop = cv2.resize(
                local_crop,
                (int(w_crop * scale), int(h_crop * scale)),
                interpolation=cv2.INTER_AREA,
            )
        elif max_dim < TARGET_MIN:
            scale = TARGET_MIN / max_dim
            local_crop = cv2.resize(
                local_crop,
                (int(w_crop * scale), int(h_crop * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        text = ""
        conf = 0.0
        # OCR kwargs that improve extraction from full trailer crops.
        # full_image_mode=True instructs the VLM to scan ALL text in the image
        # (instead of returning only the first/largest element), which is
        # critical when the small trailer number sits below the big company logo.
        # try_multiple_preprocessing=True runs several contrast/sharpness passes
        # and returns the best result.
        ocr_kwargs = {}
        is_olmocr = "OlmOCRRecognizer" in type(self.ocr).__name__
        if is_olmocr:
            ocr_kwargs["full_image_mode"] = True
            ocr_kwargs["try_multiple_preprocessing"] = True
        try:
            if self.preprocessor and self.preprocessor.enable_ocr_preprocessing:
                candidates = self.preprocessor.preprocess_for_ocr(local_crop)
                ocr_results = []
                for c in candidates:
                    result = self.ocr.recognize(c["image"], **ocr_kwargs)
                    if result.get("text", "").strip():
                        ocr_results.append(
                            {
                                "text": result.get("text", ""),
                                "conf": result.get("conf", 0.0),
                                "method": c.get("method", "unknown"),
                            }
                        )
                        if result.get("conf", 0.0) >= 0.85:
                            break
                if ocr_results:
                    best = self.preprocessor.select_best_ocr_result(ocr_results)
                    text = best.get("text", "")
                    conf = float(best.get("conf", 0.0))
            else:
                result = self.ocr.recognize(local_crop, **ocr_kwargs)
                text = result.get("text", "")
                conf = float(result.get("conf", 0.0))
        except Exception as exc:
            log.info(
                "GateVision OCR error (camera=%s track=%s): %s",
                pending.camera_id,
                pending.track_id,
                exc,
            )
        return text, conf

    def _collect_and_emit_events(self) -> None:
        """Pull any finalized fusion candidates and publish them.

        Safe to call repeatedly — fusion only releases candidates whose
        `updated_at` is older than `finalize_after_seconds`.
        """
        try:
            ready = self.fusion.collect_ready_events(now=datetime.utcnow())
        except Exception as exc:
            log.exception("[GateVision] fusion.collect_ready_events failed: %s", exc)
            return

        for event in ready:
            try:
                event = self._decorate_gate_event(event)
                self.publisher.publish(event)
                self._store_event_records(event)
                self._recent_gate_events.append(event)
                if len(self._recent_gate_events) > self._max_recent_events:
                    self._recent_gate_events = self._recent_gate_events[-self._max_recent_events :]
                self._stats["events_emitted"] += 1
                if event.get("status") == "confirmed":
                    self._stats["confirmed_events"] += 1
                else:
                    self._stats["needs_review_events"] += 1
                self._stats["last_event_at"] = event.get("ts_iso")
                if self.ingest_enabled:
                    try:
                        requests.post(self.ingest_url, json=event, timeout=1.0)
                    except Exception as exc:
                        log.info("Error posting GateVision event to ingest API: %s", exc)
            except Exception as exc:
                log.exception("[GateVision] event emission failed: %s", exc)

    def get_status(self) -> dict:
        with self._pending_obs_lock:
            pending_count = len(self._pending_obs)
        return {
            "stats": dict(self._stats),
            "recent_events": list(self._recent_gate_events),
            "config": self.get_runtime_config(),
            "active_candidates": self.fusion.get_active_candidates(),
            "offline_ocr": {
                "ocr_loaded": self.ocr is not None,
                "pending_observations": pending_count,
                "pending_max": self._pending_obs_max,
                "poll_interval_seconds": self._offline_poll_interval,
            },
        }

    def get_runtime_config(self) -> dict:
        return {
            "fusion_window_seconds": float(self.fusion.fusion_window.total_seconds()),
            "finalize_after_seconds": float(self.fusion.finalize_after.total_seconds()),
            "min_roles_required": int(self.fusion.min_roles_required),
            "ocr_run_every_n_frames": int(self.ocr_run_every_n_frames),
            "ocr_cache_max_age": int(self.ocr_cache_max_age),
            "ocr_min_confidence": float(self.ocr_min_conf),
            "max_recent_events": int(self._max_recent_events),
            "confirm_conf_threshold": float(self.fusion.confirm_conf_threshold),
            "enable_auto_close_on_departure": bool(self.enable_auto_close_on_departure),
            "enable_damage_comparison": bool(self.enable_damage_comparison),
            "offline_ocr_pending_max": int(self._pending_obs_max),
            "offline_ocr_poll_interval_seconds": float(self._offline_poll_interval),
        }

    def update_runtime_config(self, cfg: dict) -> dict:
        self._apply_gate_config(cfg or {})
        # Allow live tuning of the offline-OCR knobs without restart.
        if cfg:
            if "offline_ocr_pending_max" in cfg:
                self._pending_obs_max = max(1, int(cfg["offline_ocr_pending_max"]))
            if "offline_ocr_poll_interval_seconds" in cfg:
                self._offline_poll_interval = max(0.05, float(cfg["offline_ocr_poll_interval_seconds"]))
        return self.get_runtime_config()

    def reset_runtime_state(self) -> None:
        self._recent_gate_events = []
        self._stats = {
            "events_emitted": 0,
            "confirmed_events": 0,
            "needs_review_events": 0,
            "last_event_at": None,
        }
        self.ocr_cache.clear()
        self._event_review_overrides.clear()
        with self._pending_obs_lock:
            self._pending_obs = []

    def stop(self) -> None:
        """Cleanly shut down the offline OCR worker (called from HybridVisionPipeline.stop)."""
        self._offline_running = False
        if hasattr(self, '_stop_event') and self._stop_event is not None:
            self._stop_event.set()
        thread = getattr(self, "_offline_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(2.0, self._offline_poll_interval * 4))
            if thread.is_alive():
                log.warning("[GateVision] Offline OCR worker did not exit within timeout.")

    def review_event(self, gate_pass_id: str, decision: str, reason: str = "") -> dict:
        decision = str(decision or "").strip().lower()
        if decision not in {"confirmed", "needs_review", "rejected"}:
            return {"success": False, "message": "decision must be one of confirmed, needs_review, rejected"}
        self._event_review_overrides[gate_pass_id] = {
            "decision": decision,
            "reason": reason or "",
            "reviewed_at": datetime.utcnow().isoformat(),
        }
        for evt in self._recent_gate_events:
            if evt.get("gate_pass_id") == gate_pass_id:
                evt["status"] = decision
                evt["operator_review"] = self._event_review_overrides[gate_pass_id]
                return {"success": True, "event": evt}
        return {"success": True, "message": "review saved; event not in recent cache", "gate_pass_id": gate_pass_id}

    def _apply_gate_config(self, gate_cfg: dict) -> None:
        fusion_window_seconds = float(gate_cfg.get("fusion_window_seconds", 3.0))
        finalize_after_seconds = float(gate_cfg.get("finalize_after_seconds", 1.5))
        min_roles_required = int(gate_cfg.get("min_roles_required", 1))
        self.fusion = GateFusionEngine(
            fusion_window_seconds=fusion_window_seconds,
            finalize_after_seconds=finalize_after_seconds,
            min_roles_required=min_roles_required,
            confirm_conf_threshold=float(gate_cfg.get("confirm_conf_threshold", 0.65)),
        )
        self.ocr_run_every_n_frames = int(gate_cfg.get("ocr_run_every_n_frames", self.ocr_run_every_n_frames))
        self.ocr_cache_max_age = int(gate_cfg.get("ocr_cache_max_age", self.ocr_cache_max_age))
        self.ocr_min_conf = float(gate_cfg.get("ocr_min_confidence", self.ocr_min_conf))
        self._max_recent_events = int(gate_cfg.get("max_recent_events", self._max_recent_events))
        self.enable_auto_close_on_departure = bool(gate_cfg.get("enable_auto_close_on_departure", True))
        self.enable_damage_comparison = bool(gate_cfg.get("enable_damage_comparison", False))

    def _save_evidence_image(self, camera_id: str, gate_id: str, track_id: int, now: datetime, crop: np.ndarray) -> Optional[str]:
        try:
            gate_dir = self._evidence_dir / gate_id / camera_id
            gate_dir.mkdir(parents=True, exist_ok=True)
            name = f"{now.strftime('%Y%m%d_%H%M%S_%f')}_trk{track_id}.jpg"
            path = gate_dir / name
            ok = cv2.imwrite(str(path), crop)
            return str(path) if ok else None
        except Exception as exc:
            log.info("GateVision evidence save failed (camera=%s): %s", camera_id, exc)
            return None

    def _store_detection_record(
        self,
        gate_id: str,
        camera_id: str,
        track_id: int,
        frame_count: int,
        ts: datetime,
        text: str,
        conf: float,
        evidence_image_path: Optional[str],
        stage: str,
    ) -> None:
        if not self.db_record_sink:
            return
        try:
            self.db_record_sink(
                {
                    "gate_id": gate_id,
                    "camera_id": camera_id,
                    "track_id": track_id,
                    "frame_number": frame_count,
                    "timestamp": ts.isoformat(),
                    "trailer_id": (text or "").strip().upper() or "UNKNOWN",
                    "confidence": float(conf),
                    "image_path": evidence_image_path,
                    "stage": stage,
                }
            )
        except Exception as exc:
            log.info("GateVision DB sink failed for detection record: %s", exc)

    def _store_event_records(self, event: dict) -> None:
        if not self.db_record_sink:
            return
        try:
            for ev in event.get("evidence", []) or []:
                self.db_record_sink(
                    {
                        "gate_id": event.get("gate_id"),
                        "camera_id": ev.get("camera_id"),
                        "track_id": ev.get("track_id"),
                        "frame_number": 0,
                        "timestamp": ev.get("ts_iso") or event.get("ts_iso"),
                        "trailer_id": event.get("trailer_id") or "UNKNOWN",
                        "confidence": float(ev.get("conf") or event.get("conf") or 0.0),
                        "image_path": ev.get("evidence_image_path"),
                        "stage": event.get("event_type") or "gate_event",
                    }
                )
        except Exception as exc:
            log.info("GateVision DB sink failed for fused event record: %s", exc)

    def _decorate_gate_event(self, event: dict) -> dict:
        # Convert direction into operational arrival/departure semantics.
        direction = str(event.get("direction", "unknown")).lower()
        lifecycle_stage = "unknown"
        if direction == "entry":
            lifecycle_stage = "arrival"
        elif direction == "exit":
            lifecycle_stage = "departure"
        event["lifecycle_stage"] = lifecycle_stage
        event["event_type"] = f"gate_{lifecycle_stage}" if lifecycle_stage != "unknown" else "gate_pass"

        trailer_id, scac = self._extract_trailer_and_scac(event.get("trailer_id"))
        event["trailer_id"] = trailer_id
        event["scac"] = scac

        if lifecycle_stage == "departure":
            event["workflow_action"] = "close_workflow" if (self.enable_auto_close_on_departure and event.get("status") == "confirmed") else "await_review"
            event["close_workflow"] = bool(self.enable_auto_close_on_departure and event.get("status") == "confirmed")
            event["departure_verified"] = event.get("status") == "confirmed"
        else:
            event["arrival_candidate"] = True
            event["arrival_confirmed"] = event.get("status") == "confirmed"

        event["damage_comparison"] = {
            "enabled": self.enable_damage_comparison,
            "status": "pending" if self.enable_damage_comparison else "not_enabled",
        }

        app = getattr(self.metrics_server, "frame_storage", None)
        event["detection_mode"] = getattr(app, "detection_mode", "trailer") if app else "trailer"

        override = self._event_review_overrides.get(event.get("gate_pass_id"))
        if override:
            event["status"] = override["decision"]
            event["operator_review"] = override
        return event

    @staticmethod
    def _get_scac_from_brand(text: str) -> Optional[str]:
        t = (text or "").strip().upper()
        if not t:
            return None
        # Check J.B. Hunt keywords (including Intermodal/Chassis prefix IBHU)
        if "J.B. HUNT" in t or "JB HUNT" in t or "JBHUNT" in t or "JBHU" in t or "HUNT" in t or "IBHU" in t:
            return "JBHU"
        if "SCHNEIDER" in t or "SNLU" in t:
            return "SNLU"
        if "SWIFT" in t or "SWFT" in t:
            return "SWFT"
        if "HUB GROUP" in t or "HGIU" in t:
            return "HGIU"
        if "KNIGHT" in t or "KNIG" in t:
            return "KNIG"
        if "AMAZON" in t or "AZNG" in t:
            return "AZNG"
        if "XPO" in t:
            return "XPO"
        if "FEDEX" in t or "FXFE" in t:
            return "FXFE"
        if "UPS" in t or "UPSS" in t:
            return "UPSS"
        return None

    @staticmethod
    def _extract_trailer_and_scac(text: Optional[str]) -> tuple:
        raw = (text or "").strip().upper()
        if not raw:
            return None, None

        # Clean the raw text by keeping letters, digits, and spaces
        # If the text is contiguous letters followed by digits (e.g. "IBHU322099" or "JBHUIBHU322099"), split directly
        clean_raw = re.sub(r"[^A-Z0-9 ]", "", raw)
        m = re.match(r"^([A-Z\s]+)\s*(\d+)$", clean_raw)
        if m:
            letters = re.sub(r"\s", "", m.group(1))
            trailer = m.group(2)
            if len(letters) >= 2:
                scac = None
                if len(letters) <= 4:
                    scac = letters
                brand_scac = GateVisionPipeline._get_scac_from_brand(raw)
                if brand_scac:
                    scac = brand_scac
                return trailer, scac
            else:
                return re.sub(r"\s", "", clean_raw), None

        # Reconstruct vertical trailer numbers (e.g. 'R\n5\n3\n2\n7\n5' or 'R 5 3 2 7 5')
        # Split raw text into characters/words
        initial_parts = re.split(r"[\s\-_]+", raw)
        parts = []
        temp_vertical = []
        
        for p in initial_parts:
            # If the part is a single character (letter or digit), buffer it
            if len(p) == 1 and p.isalnum():
                temp_vertical.append(p)
            else:
                # If we were buffering a vertical number, join and add it first
                if temp_vertical:
                    parts.append("".join(temp_vertical))
                    temp_vertical = []
                parts.append(p)
        # Flush any remaining vertical number buffer
        if temp_vertical:
            parts.append("".join(temp_vertical))

        scac = None
        
        # We rebuild the full cleaned string from the reconstructed parts to check for digits
        rebuilt_raw = " ".join(parts)
        has_digits = bool(re.search(r"\d", rebuilt_raw))
        
        # Define candidate scoring function to prioritize real trailer IDs and penalize brand/model names
        def score_candidate(cand: str) -> float:
            cand_up = cand.strip().upper()
            ignore_patterns = {
                "3000R", "4000D", "2000A", "CARRIER", "THERMOKING", "THERMO_KING", 
                "UTILITY", "GREATDANE", "GREAT_DANE", "REEFER", "SUPERII", "TK"
            }
            if cand_up in ignore_patterns:
                return -10.0
                
            digits = [c for c in cand_up if c.isdigit()]
            letters = [c for c in cand_up if c.isalpha()]
            
            if not digits:
                return -1.0
                
            num_digits = len(digits)
            num_letters = len(letters)
            
            # 5 to 6 digits only (perfect trailer ID)
            if num_digits in (5, 6) and num_letters == 0:
                return 10.0
                
            # Letter prefix + 5 to 6 digits (perfect SCAC/reefer trailer ID)
            if num_digits in (5, 6) and num_letters in (1, 2, 3, 4):
                return 9.0
                
            # 4 digits only (very common trailer ID)
            if num_digits == 4 and num_letters == 0:
                return 8.0
                
            # 4 digits + letter prefix
            if num_digits == 4 and num_letters in (1, 2, 3, 4):
                return 7.0
                
            # 3 digits only
            if num_digits == 3 and num_letters == 0:
                return 4.0
                
            return 1.0

        best_trailer = None
        best_score = -999.0

        for p in parts:
            if has_digits and len(p) == 4 and p.isalpha() and scac is None:
                scac = p
            if re.search(r"\d", p):
                score = score_candidate(p)
                if score > best_score:
                    best_score = score
                    best_trailer = p
                    
        trailer = best_trailer
        if trailer is None:
            trailer = rebuilt_raw
        else:
            # Run the split regex on the best_trailer candidate itself
            # to separate any joined SCAC prefix (e.g. "IBHU322099" or "JBHUIBHU322099")
            clean_cand = re.sub(r"[^A-Z0-9 ]", "", trailer)
            m = re.match(r"^([A-Z\s]+)\s*(\d+)$", clean_cand)
            if m:
                letters = re.sub(r"\s", "", m.group(1))
                digits = m.group(2)
                if len(letters) >= 2:
                    trailer = digits
                    cand_scac = None
                    if len(letters) <= 4:
                        cand_scac = letters
                    brand_scac = GateVisionPipeline._get_scac_from_brand(raw)
                    if brand_scac:
                        cand_scac = brand_scac
                    if scac is None:
                        scac = cand_scac

        # Assign or override SCAC if a carrier brand name is detected
        brand_scac = GateVisionPipeline._get_scac_from_brand(raw)
        if brand_scac:
            scac = brand_scac

        return trailer, scac

