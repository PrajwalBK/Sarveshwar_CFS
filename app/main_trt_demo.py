"""
Main Application Loop - Trailer Vision Edge

End-to-end processing pipeline:
1. Camera ingestion (RTSP/USB)
2. Detection (YOLO TensorRT)
3. Tracking (ByteTrack)
4. OCR (TrOCR/PaddleOCR/CRNN TensorRT)
5. GPS coordinates from GPS sensor/log files
6. Spot resolution (GeoJSON polygons)
7. Logging, publishing, metrics
"""

import os
import sys
import yaml
import json
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import signal
import threading
import gc
import time

# Try to import torch for GPU memory management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Application modules
from app.rtsp import open_stream, frame_generator
from app.ai.tracker_bytetrack import ByteTrackWrapper
from app.ocr.recognize import PlateRecognizer
from app.csv_logger import CSVLogger
from app.media_rotator import MediaRotator
from app.event_bus import MultiPublisher
from app.uploader import UploadManager
from app.metrics_server import MetricsServer
from app.video_processor import VideoProcessor
from app.image_preprocessing import ImagePreprocessor
from app.video_recorder import VideoRecorder
from app.processing_queue import ProcessingQueueManager
from app.video_frame_db import VideoFrameDB
from app.app_logger import setup_logging, get_logger
from app.pipelines import create_pipeline
from app.detection_runtime import build_live_detector, expected_detection_class
import requests

log = get_logger(__name__)


class TrailerVisionApp:
    """
    Main application class for trailer vision processing.
    """
    
    def __init__(self, config_path: str = "config/cameras.yaml"):
        """
        Initialize application.
        
        Args:
            config_path: Path to cameras.yaml configuration
        """
        setup_logging()
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False
        self.pipeline_mode = "hybrid"  # set from config in _initialize_components
        
        # Initialize components
        self.detector = None
        self.ocr = None
        self.spot_resolver = None
        self.csv_logger = None
        self.media_rotator = None
        self.publisher = None
        self.uploader = None
        self.metrics_server = None
        self.preprocessor = None
        self.processing_queue = None  # Processing queue manager for automated workflow
        self.video_frame_db = None  # Database for storing video frame records
        self.pipeline = None
        
        # Per-camera trackers
        self.trackers = {}
        self.camera_3d_projectors = {}  # camera_id -> Camera3DProjector (priority)
        
        # Check if OCR is oLmOCR (needs GPU memory cleanup)
        self.is_olmocr = False  # Will be set after OCR initialization

        # Set True while a multi-GB VLM is being loaded into GPU. The MJPEG live
        # streamer in metrics_server checks this and stalls so its concurrent
        # JPEG-encode buffers don't OOM-kill the loader on Jetson Orin.
        self.gpu_load_in_progress = False

        # Frame storage for camera feed display (thread-safe)
        import threading
        self.frame_lock = threading.Lock()
        self.latest_frames = {}  # camera_id -> latest frame
        
        # Metrics tracking
        self.camera_metrics = {}
        
        # Frame storage for video streaming (thread-safe)
        self.latest_frames = {}
        self.frame_lock = threading.Lock()
        self.detector_runtime_lock = threading.Lock()
        
        # OCR optimization: Cache results per track_id to avoid re-processing
        self.ocr_cache = {}  # (camera_id, track_id) -> {'text': str, 'conf': float, 'frame': int, 'last_updated': int}
        self.ocr_cache_max_age = 30  # Re-run OCR every 30 frames if no good result
        self.ocr_min_confidence = 0.5  # Only cache results with confidence >= this
        self.ocr_run_every_n_frames = 10  # Run OCR every N frames for existing tracks (if not cached)
        self.ocr_cache_max_size = 100  # Maximum cache entries to prevent memory leaks
        
        # GPS sensor and video recorder
        self.gps_sensor = None
        self.video_recorder = None
        
        # Graceful shutdown state tracking
        self.recording_stopped = False  # True when recording stopped but processing continues
        self.shutdown_lock = threading.Lock()  # Lock for thread-safe state access
        # Serialize OCR load: startup backlog thread vs /api/start-application vs debug routes
        self._ocr_init_lock = threading.Lock()

        # Detection mode for automatic pipeline (car vs trailer); can be overridden in config
        self.detection_mode = 'trailer'
        
        # Runtime recordings watcher (auto-queue manually copied videos while app is running)
        self._recordings_watcher_thread = None
        self._recordings_watcher_stop = threading.Event()
        self._queued_video_signatures = set()  # {(abs_path, mtime_ns, size_bytes)}
        self._pending_video_signatures = {}  # abs_path -> {'sig': tuple, 'first_seen': epoch_s}
        self._watcher_scan_interval_seconds = 3.0
        self._watcher_settle_seconds = 2.0
        self._initialize_components()
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _initialize_components(self):
        """Initialize all application components."""
        globals_cfg = self.config.get('globals', {})
        # Detection mode: .env DETECTION_MODE overrides config, default 'trailer' (used by startup mode and recording)
        dm = os.getenv('DETECTION_MODE') or globals_cfg.get('detection_mode') or 'gate'
        dm = str(dm).strip().lower()
        self.detection_mode = dm if dm in ('gate', 'stacker') else 'gate'
        log.info(f"[TrailerVisionApp] Detection mode: {self.detection_mode} (from env DETECTION_MODE or config)")
        
        self.detector = build_live_detector(self.detection_mode, globals_cfg)
        if self.detector is None:
            log.warning("No detector available for mode=%s (see detection_runtime logs)", self.detection_mode)
        
        # Initialize OCR - Lazy loading: Don't load OCR initially to save memory
        # OCR will be loaded on-demand when explicitly requested (not automatically)
        # This prevents memory issues when processing large videos
        self.ocr = None
        self.is_olmocr = False
        log.info(f"[TrailerVisionApp] OCR loading deferred - will be loaded on-demand when requested")
        
        # Spot resolver not used; parking spots are loaded from all config/*.csv in the data processor.
        # self.spot_resolver remains None.

        # Initialize uploader
        self.uploader = UploadManager()
        
        # Initialize CSV logger
        self.csv_logger = CSVLogger(uploader=self.uploader)
        
        # Initialize media rotator
        keep_screenshots = globals_cfg.get('keep_screenshots', 10)
        self.media_rotator = MediaRotator(
            keep_last=keep_screenshots,
            uploader=self.uploader
        )
        
        # Initialize event publisher
        self.publisher = MultiPublisher()
        
        # REST ingest client (if enabled)
        self.ingest_enabled = os.getenv('INGEST_ENABLED', 'false').lower() == 'true'
        self.ingest_url = os.getenv('INGEST_URL', 'http://localhost:8000/events')
        
        # Initialize metrics server
        metrics_port = int(os.getenv('METRICS_PORT', '8080'))
        self.metrics_server = MetricsServer(
            port=metrics_port, 
            csv_logger=self.csv_logger,
            frame_storage=self  # Pass self for camera feed display
        )
        self.metrics_server.start()
        
        # Gate Vision uses camera OCR & video feeds without requiring a GPS sensor
        self.gps_sensor = None
        
        # Initialize video recorder with 45-second auto-chunking
        # Callback will be set after processing queue is initialized
        self.video_recorder = VideoRecorder(
            output_dir="out/recordings",
            gps_sensor=self.gps_sensor,
            chunk_duration_seconds=30.0,  # 30-second chunks
            on_chunk_saved=None  # Will be set after processing queue initialization
        )
        log.info(f"[TrailerVisionApp] Video recorder initialized with 45-second auto-chunking")
        
        # Initialize image preprocessor
        # IMPORTANT: Disable rotation for oLmOCR and EasyOCR because they handle rotation internally.
        # Pre-rotating images causes double rotation issues.
        preproc_cfg = globals_cfg.get('preprocessing', {})
        ocr_type = str(type(self.ocr)) if self.ocr is not None else ""
        is_olmocr = 'OlmOCRRecognizer' in ocr_type
        is_easyocr = 'EasyOCRRecognizer' in ocr_type
        # Disable rotation for oLmOCR and EasyOCR (both handle rotation internally)
        enable_rotation = not (is_olmocr or is_easyocr)
        
        self.preprocessor = ImagePreprocessor(
            enable_yolo_preprocessing=preproc_cfg.get('enable_yolo', True),
            enable_ocr_preprocessing=preproc_cfg.get('enable_ocr', True),
            yolo_strategy=preproc_cfg.get('yolo_strategy', 'enhanced'),
            ocr_strategy=preproc_cfg.get('ocr_strategy', 'multi'),
            enable_rotation=enable_rotation
        )
        log.info(f"[TrailerVisionApp] Image preprocessing enabled:")
        log.info(f"  YOLO: {self.preprocessor.enable_yolo_preprocessing} ({self.preprocessor.yolo_strategy})")
        log.info(f"  OCR: {self.preprocessor.enable_ocr_preprocessing} ({self.preprocessor.ocr_strategy})")
        if is_olmocr:
            rotation_msg = "Disabled (oLmOCR handles rotation internally)"
        elif is_easyocr:
            rotation_msg = "Disabled (EasyOCR handles rotation internally)"
        else:
            rotation_msg = "Enabled"
        log.info(f"  Rotation: {rotation_msg}")
        
        # Load 3D projectors for each camera (GPS coordinates from GPS sensor only)
        for camera in self.config.get('cameras', []):
            camera_id = camera['id']
            
            pass
        
        # Initialize trackers for each camera
        # Use lower track threshold to match detector confidence
        globals_cfg = self.config.get('globals', {})
        track_thresh = globals_cfg.get('detector_conf', 0.20)
        
        for camera in self.config.get('cameras', []):
            camera_id = camera['id']
            self.trackers[camera_id] = ByteTrackWrapper(track_thresh=track_thresh)
            self.camera_metrics[camera_id] = {
                'frames_processed': 0,
                'fps_ema': 0.0,
                'last_publish': None
            }

        # Initialize pluggable vision pipeline (yard / gate / hybrid from globals.vision_pipeline or VISION_PIPELINE)
        self.pipeline = create_pipeline(self)
        self.pipeline_mode = getattr(self.pipeline, "vision_mode", "hybrid")
        log.info(
            "[TrailerVisionApp] Vision pipeline: %s (%s)",
            self.pipeline_mode,
            self.pipeline.name,
        )
        
        # Initialize video processor for testing
        def create_tracker():
            return ByteTrackWrapper(track_thresh=track_thresh)
        
        # GPS coordinates from GPS sensor only (no static calibration reference)
        cameras_list = self.config.get('cameras', [])
        log.info(f"[TrailerVisionApp] Initializing video processor:")
        log.info(f"  - Detector: {'Available' if self.detector else 'NOT AVAILABLE'}")
        log.info(f"  - OCR: {'Available' if self.ocr else 'NOT AVAILABLE'}")
        log.info(f"  - Spot Resolver: {'Available' if self.spot_resolver else 'NOT AVAILABLE'}")
        # Check for 3D projector
        test_3d_projector = None
        if cameras_list:
            first_camera_id = cameras_list[0]['id']
            if first_camera_id in self.camera_3d_projectors:
                test_3d_projector = self.camera_3d_projectors[first_camera_id]
        
        log.info(f"  - 3D Projector: {'Available' if test_3d_projector is not None else 'NOT AVAILABLE'}")
        log.info(f"  - GPS: GPS sensor only")
        try:
            # Determine camera_id for homography loading (use first camera or "test-video")
            test_camera_id = "test-video"
            if cameras_list:
                test_camera_id = cameras_list[0]['id']
            
            self.video_processor = VideoProcessor(
                preprocessor=self.preprocessor,
                detector=self.detector,
                ocr=self.ocr,
                tracker_factory=create_tracker,
                spot_resolver=self.spot_resolver,
                bev_projector=None,
                gps_reference=None,  # GPS from sensor / gps_log only
                camera_id=test_camera_id
            )
            log.info(f"[TrailerVisionApp] Video processor created successfully")
        except Exception as e:
            log.error("[TrailerVisionApp] Failed to create video processor: %s", e)
            import traceback
            traceback.print_exc()
            self.video_processor = None
        
        # Update metrics server with video processor and frame storage
        if self.metrics_server:
            self.metrics_server.video_processor = self.video_processor
            self.metrics_server.frame_storage = self
            log.info(f"[TrailerVisionApp] Video processor assigned to metrics server: {self.metrics_server.video_processor is not None}")
        else:
            log.error("[TrailerVisionApp] Metrics server not initialized!")
        
        # Processing queue will be initialized when Start Application is called
        # This allows lazy loading of OCR to save memory until needed
        self.processing_queue = None
        log.info(f"[TrailerVisionApp] Processing queue will be initialized when application starts")
        
        # Initialize database for video frame records
        try:
            self.video_frame_db = VideoFrameDB(db_path="data/video_frames.db")
            log.info(f"[TrailerVisionApp] Video frame database initialized")
        except Exception as e:
            log.warning("[TrailerVisionApp] Failed to initialize database: %s", e)
            self.video_frame_db = None
        
        # Periodic upload to Prosper Smart Yard (YardVision processed rows + GateVision fused rows); delete from SQLite after success.
        #
        # The upload loop now runs in ALL pipeline modes (yard / hybrid / gate)
        # so live gate-pass events emitted by GateVisionPipeline get pushed to
        # Prosper continuously without the operator having to click "Upload
        # pending to Prosper". The upload function itself fetches YardVision
        # rows + GateVision rows separately:
        #   * gate-only mode: only GateVision rows exist → yard query returns
        #     [] → upload only sends gate-events. No accidental YardVision
        #     traffic.
        #   * yard / hybrid: both tables can have rows → both are sent.
        # Earlier this loop was disabled in gate mode; that was over-correction
        # for "no YardVision uploads in gate mode" — it suppressed gate uploads
        # too and forced operators to click the manual button. Set
        # ``EDGE_DISABLE_BACKGROUND_UPLOAD=1`` to disable the loop entirely.
        self._upload_thread = None
        self._upload_thread_stop = threading.Event()
        self._upload_status_lock = threading.Lock()
        prosper_ready = self._prosper_upload_credentials_ready()
        upload_loop_disabled_env = os.getenv("EDGE_DISABLE_BACKGROUND_UPLOAD", "").lower() in (
            "1", "true", "yes", "on",
        )
        upload_enabled_for_mode = not upload_loop_disabled_env  # all pipeline modes welcome
        self.upload_status = {
            "enabled": prosper_ready and upload_enabled_for_mode,
            "is_uploading": False,
            "last_run_at": None,
            "config_message": (
                "" if prosper_ready and upload_enabled_for_mode
                else (
                    "disabled via EDGE_DISABLE_BACKGROUND_UPLOAD env"
                    if not upload_enabled_for_mode
                    else self._prosper_upload_config_message()
                )
            ),
            "yard": self._empty_prosper_branch_status(),
            "gate": self._empty_prosper_branch_status(),
        }
        if self.video_frame_db and upload_enabled_for_mode:
            self._upload_thread = threading.Thread(target=self._upload_processed_loop, daemon=True, name="UploadProcessedRecords")
            self._upload_thread.start()
            interval = max(30, int(os.getenv("EDGE_UPLOAD_INTERVAL_SECONDS", "60")))
            log.info(
                "[TrailerVisionApp] Prosper upload thread started (interval %ss, credentials_ok=%s, pipeline_mode=%s)",
                interval,
                prosper_ready,
                self.pipeline_mode,
            )
        elif self.video_frame_db and not upload_enabled_for_mode:
            log.info(
                "[TrailerVisionApp] Prosper upload thread NOT started: "
                "EDGE_DISABLE_BACKGROUND_UPLOAD is set."
            )
        
        # Startup mode: load assets and process all unprocessed videos in recordings folder.
        # In `gate` pipeline mode this is OFF by default — GateVisionPipeline handles live
        # detect→crop→OCR per-vehicle. Backlog processing is reserved for the dashboard button
        # (POST /api/process-backlog) for replay/testing. Other modes (yard/hybrid) keep the
        # original auto-on-startup behavior.
        self._startup_backlog_lock = threading.Lock()

        # Pre-warm OCR model only if explicitly requested via PRELOAD_OCR=1 (otherwise load on demand)
        if os.getenv("PRELOAD_OCR", "0").lower() in ("1", "true", "yes"):
            try:
                log.info("[TrailerVisionApp] Pre-loading OCR model into GPU VRAM during app startup...")
                self._initialize_ocr()
            except Exception as ocr_init_err:
                log.warning("[TrailerVisionApp] Startup OCR pre-warm deferred: %s", ocr_init_err)
        else:
            log.info("[TrailerVisionApp] OCR will load on-demand when crops are queued (saving RAM).")
        
        self._startup_backlog_status = {
            'status': 'pending',  # pending | loading_assets | queueing | done | error
            'message': 'Waiting to start...',
            'videos_queued': 0,
            'error': None
        }
        # Automatic scanning of out/recordings on startup is DISABLED by default
        disable_startup_backlog = os.getenv("DISABLE_STARTUP_BACKLOG", "1").lower() not in ("0", "false", "no", "off")
        if disable_startup_backlog:
            log.info("[TrailerVisionApp] Startup backlog video processing disabled by default.")
        else:
            log.info("[TrailerVisionApp] Startup mode: will load assets and queue backlog videos from recordings folder")
            _startup_thread = threading.Thread(target=self._run_startup_backlog_processing, daemon=True, name="StartupBacklog")
            _startup_thread.start()
    
    def get_startup_backlog_status(self) -> Dict:
        """Return current startup mode status (thread-safe)."""
        with self._startup_backlog_lock:
            return dict(self._startup_backlog_status)
    
    def _run_startup_backlog_processing(self):
        """Background: load assets then queue all videos in out/recordings for processing (startup mode)."""
        try:
            time.sleep(2)  # Let app and server fully start
            with self._startup_backlog_lock:
                self._startup_backlog_status['status'] = 'loading_assets'
                self._startup_backlog_status['message'] = 'Loading OCR and initializing processing queue...'
                self._startup_backlog_status['error'] = None
            log.info("[TrailerVisionApp] Startup mode: initializing assets...")
            result = self.initialize_assets()
            if not result.get('success'):
                msg = result.get('message', 'Asset initialization failed')
                with self._startup_backlog_lock:
                    self._startup_backlog_status['status'] = 'error'
                    self._startup_backlog_status['message'] = msg
                    self._startup_backlog_status['error'] = msg
                log.warning("[TrailerVisionApp] Startup mode: asset initialization failed: %s", msg)
                return
            if self.pipeline_mode == "gate":
                with self._startup_backlog_lock:
                    self._startup_backlog_status['status'] = 'done'
                    self._startup_backlog_status['message'] = 'Live GateVision OCR assets initialized.'
                    self._startup_backlog_status['videos_queued'] = 0
                    self._startup_backlog_status['error'] = None
                log.info("[TrailerVisionApp] Startup mode: initialized live GateVision OCR assets (skipped backlog queueing)")
                return

            with self._startup_backlog_lock:
                self._startup_backlog_status['status'] = 'queueing'
                self._startup_backlog_status['message'] = 'Scanning recordings folder and queueing videos...'
            queued = self._queue_backlog_videos()
            with self._startup_backlog_lock:
                self._startup_backlog_status['status'] = 'done'
                self._startup_backlog_status['message'] = f'Queued {queued} video(s). Processing in background.' if queued else 'No videos found in recordings folder.'
                self._startup_backlog_status['videos_queued'] = queued
                self._startup_backlog_status['error'] = None
            log.info("[TrailerVisionApp] Startup mode: queued %d video(s) from recordings folder", queued)
        except Exception as e:
            with self._startup_backlog_lock:
                self._startup_backlog_status['status'] = 'error'
                self._startup_backlog_status['message'] = str(e)
                self._startup_backlog_status['error'] = str(e)
            log.exception("[TrailerVisionApp] Startup mode error: %s", e)
    
    def _queue_backlog_videos(self) -> int:
        """Scan out/recordings (or out/recording) and queue every video for processing. Returns count queued."""
        if not self.processing_queue:
            return 0
        recordings_dir = None
        for folder_name in ["out/recordings", "out/recording"]:
            test_dir = Path(folder_name)
            if test_dir.exists():
                recordings_dir = test_dir
                break
        if not recordings_dir or not recordings_dir.exists():
            log.info("[TrailerVisionApp] Startup mode: no recordings directory found")
            return 0
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(recordings_dir.glob(f'*{ext}'))
            video_files.extend(recordings_dir.rglob(f'*{ext}'))
        video_files = list(set(video_files))
        if not video_files:
            log.info("[TrailerVisionApp] Startup mode: no video files in %s", recordings_dir)
            return 0
        globals_cfg = self.config.get('globals', {})
        detect_every_n = globals_cfg.get('detect_every_n', 5)
        detection_mode = getattr(self, 'detection_mode', 'trailer')
        queued_count = 0
        for vid_path in sorted(video_files):
            try:
                if self._queue_video_for_processing(vid_path, recordings_dir, detect_every_n, detection_mode):
                    queued_count += 1
                    log.info("[TrailerVisionApp] Startup mode: queued %s", vid_path.name)
            except Exception as e:
                log.warning("[TrailerVisionApp] Startup mode: failed to queue %s: %s", vid_path.name, e)
        return queued_count
    
    def _resolve_video_gps_log(self, video_path: Path, recordings_root: Path) -> Optional[str]:
        """Find matching GPS log for a video path."""
        video_stem = video_path.stem
        video_dir = video_path.parent
        for gps_path in [
            video_dir / f"{video_stem}.json",
            video_dir / f"{video_stem}_gps.json",
            recordings_root / f"{video_stem}.json",
            recordings_root / f"{video_stem}_gps.json",
        ]:
            if gps_path.exists():
                return str(gps_path)
        return None
    
    def _queue_video_for_processing(self, video_path: Path, recordings_root: Path, detect_every_n: int, detection_mode: str) -> bool:
        """
        Queue a single video for processing if not already queued for this file signature.
        Returns True if queued, False if skipped.
        """
        if not self.processing_queue:
            return False
        try:
            resolved = video_path.resolve()
            stat = resolved.stat()
            signature = (str(resolved), stat.st_mtime_ns, stat.st_size)
        except Exception:
            return False
        if signature in self._queued_video_signatures:
            return False
        gps_log = self._resolve_video_gps_log(video_path, recordings_root)
        parts = video_path.stem.split('_')
        vid_camera_id = parts[0] if parts else "test-video"
        self.processing_queue.queue_video_processing(
            video_path=str(video_path),
            camera_id=vid_camera_id,
            gps_log_path=gps_log,
            detect_every_n=detect_every_n,
            detection_mode=detection_mode
        )
        self._queued_video_signatures.add(signature)
        return True
    
    def _start_recordings_watcher(self):
        """Start background watcher that auto-queues new videos copied to recordings folders."""
        if self._recordings_watcher_thread and self._recordings_watcher_thread.is_alive():
            return
        self._recordings_watcher_stop.clear()
        self._recordings_watcher_thread = threading.Thread(
            target=self._recordings_watcher_loop,
            daemon=True,
            name="RecordingsWatcher"
        )
        self._recordings_watcher_thread.start()
        log.info("[TrailerVisionApp] Recordings watcher started (scan interval %.1fs)", self._watcher_scan_interval_seconds)
    
    def _recordings_watcher_loop(self):
        """Poll recordings folders and queue newly added settled video files."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}
        while not self._recordings_watcher_stop.wait(timeout=self._watcher_scan_interval_seconds):
            if not self.running or not self.processing_queue:
                continue
            globals_cfg = self.config.get('globals', {})
            detect_every_n = globals_cfg.get('detect_every_n', 5)
            detection_mode = getattr(self, 'detection_mode', 'trailer')
            recordings_dir = None
            for folder_name in ["out/recordings", "out/recording"]:
                d = Path(folder_name)
                if d.exists():
                    recordings_dir = d
                    break
            if not recordings_dir:
                continue
            try:
                candidates = []
                for p in recordings_dir.rglob('*'):
                    if p.is_file() and p.suffix.lower() in video_extensions:
                        candidates.append(p)
                now_ts = time.time()
                for vid_path in sorted(candidates):
                    try:
                        resolved = vid_path.resolve()
                        stat = resolved.stat()
                        sig = (str(resolved), stat.st_mtime_ns, stat.st_size)
                        if sig in self._queued_video_signatures:
                            continue
                        if now_ts - stat.st_mtime < self._watcher_settle_seconds:
                            continue
                        key = str(resolved)
                        pending = self._pending_video_signatures.get(key)
                        if not pending or pending.get('sig') != sig:
                            self._pending_video_signatures[key] = {'sig': sig, 'first_seen': now_ts}
                            continue
                        if now_ts - pending.get('first_seen', now_ts) < self._watcher_settle_seconds:
                            continue
                        if self._queue_video_for_processing(vid_path, recordings_dir, detect_every_n, detection_mode):
                            log.info("[TrailerVisionApp] Watcher queued new video: %s", vid_path.name)
                        self._pending_video_signatures.pop(key, None)
                    except Exception as e:
                        log.debug("[TrailerVisionApp] Watcher skipped %s: %s", vid_path, e)
            except Exception as e:
                log.warning("[TrailerVisionApp] Recordings watcher scan error: %s", e)
    
    def _prosper_upload_credentials_ready(self) -> bool:
        """True when Prosper site id and auth env are set so uploads can succeed.

        Accepts EITHER an explicit API key / pre-issued bearer token (legacy)
        OR a full set of login credentials (PROSPER_SITE_CODE, PROSPER_EMAIL,
        PROSPER_PASSWORD) so the edge can self-refresh tokens via
        ``POST /api/auth/login``.
        """
        if not self.video_frame_db:
            return False
        from app.container_utils import prosper_site_id_valid

        globals_cfg = self.config.get("globals", {}) or {}
        site_id = (os.getenv("PROSPER_SITE_ID") or globals_cfg.get("prosper_site_id") or "").strip()
        if not prosper_site_id_valid(site_id):
            return False
        if (
            os.getenv("DASHBOARD_API_KEY")
            or os.getenv("PROSPER_API_KEY")
            or os.getenv("PROSPER_BEARER_TOKEN")
        ):
            return True
        # New: auto-refresh path needs site code + email + password (passwords
        # are intentionally env-only — never read from config files).
        site_code = os.getenv("PROSPER_SITE_CODE") or globals_cfg.get("prosper_site_code")
        email = os.getenv("PROSPER_EMAIL") or globals_cfg.get("prosper_email")
        password = os.getenv("PROSPER_PASSWORD")
        return bool(site_code and email and password)

    @staticmethod
    def _empty_prosper_branch_status() -> Dict:
        return {
            "last_result": None,
            "last_batch_count": 0,
            "last_deleted_count": 0,
            "last_error": None,
            "total_uploaded": 0,
            "last_response_status": None,
            "last_response_body": None,
        }

    def _prosper_upload_config_message(self) -> str:
        if not self.video_frame_db:
            return "Video frame database not available."
        from app.container_utils import prosper_site_id_valid

        globals_cfg = self.config.get("globals", {}) or {}
        site_id = (os.getenv("PROSPER_SITE_ID") or globals_cfg.get("prosper_site_id") or "").strip()
        if not prosper_site_id_valid(site_id):
            return "Set PROSPER_SITE_ID (UUID) or globals.prosper_site_id in config/cameras.yaml."
        has_legacy_auth = bool(
            os.getenv("DASHBOARD_API_KEY")
            or os.getenv("PROSPER_API_KEY")
            or os.getenv("PROSPER_BEARER_TOKEN")
        )
        site_code = os.getenv("PROSPER_SITE_CODE") or globals_cfg.get("prosper_site_code")
        email = os.getenv("PROSPER_EMAIL") or globals_cfg.get("prosper_email")
        password = os.getenv("PROSPER_PASSWORD")
        has_login = bool(site_code and email and password)
        if not has_legacy_auth and not has_login:
            return (
                "Set Prosper auth: either PROSPER_BEARER_TOKEN (legacy, expires) "
                "OR set PROSPER_SITE_CODE + PROSPER_EMAIL + PROSPER_PASSWORD so the "
                "edge can self-authenticate via POST /api/auth/login and refresh "
                "automatically on 401."
            )
        return ""

    def _upload_processed_loop(self):
        """Background loop: upload to Prosper and delete from SQLite after success."""
        interval = max(30, int(os.getenv("EDGE_UPLOAD_INTERVAL_SECONDS", "60")))
        while not self._upload_thread_stop.wait(timeout=interval):
            if not self.running:
                continue
            try:
                self.upload_processed_records_and_delete()
            except Exception as e:
                log.warning("[TrailerVisionApp] Upload processed records error: %s", e)

    def upload_processed_records_and_delete(self):
        """
        Upload to Prosper Smart Yard (http://syapi.prosperassettracking.com/swagger/):
        - YardVision: processed SQLite rows (is_processed=1) → POST /api/sites/{siteId}/locations/trailers
        - GateVision: fused gatevision:* rows → POST /api/sites/{siteId}/gate-events
        Then delete successful rows from SQLite.
        """
        if not self.video_frame_db:
            return
        globals_cfg = self.config.get("globals", {}) or {}
        self._upload_processed_records_prosper(globals_cfg)

    def _get_prosper_auth(self, base_url: str, globals_cfg: Dict):
        """Lazy-construct & cache a ``ProsperAuth`` provider tied to this app.

        Re-uses the same instance across upload iterations so the cached JWT
        survives between background-thread ticks. Returns None if the
        ``app.prosper_auth`` module isn't importable for any reason (then
        upload falls back to the legacy static-token path).
        """
        try:
            from app.prosper_auth import ProsperAuth
        except Exception as e:
            log.warning("[TrailerVisionApp] prosper_auth import failed: %s", e)
            return None
        existing = getattr(self, "_prosper_auth", None)
        if existing is not None and getattr(existing, "base_url", None) == base_url.rstrip("/"):
            return existing
        auth = ProsperAuth.from_env(base_url=base_url, globals_cfg=globals_cfg)
        self._prosper_auth = auth
        return auth

    def _upload_processed_records_prosper(self, globals_cfg: Dict) -> None:
        """Upload YardVision trailer locations and GateVision gate events to Prosper."""
        from app.prosper_gate_upload import upload_gatevision_records_prosper
        from app.container_utils import (
            prosper_site_id_valid,
            upload_processed_yard_records_prosper,
        )

        base_url = (
            os.getenv("PROSPER_API_BASE_URL")
            or globals_cfg.get("prosper_api_base_url")
            or "http://syapi.prosperassettracking.com"
        ).rstrip("/")
        site_id = (os.getenv("PROSPER_SITE_ID") or globals_cfg.get("prosper_site_id") or "").strip()
        if not prosper_site_id_valid(site_id):
            log.warning(
                "[TrailerVisionApp] Prosper upload skipped: set PROSPER_SITE_ID or globals.prosper_site_id (UUID)."
            )
            err = "missing or invalid PROSPER_SITE_ID"
            with self._upload_status_lock:
                self.upload_status["last_run_at"] = datetime.utcnow().isoformat() + "Z"
                for key in ("yard", "gate"):
                    b = self.upload_status[key]
                    b["last_result"] = "skipped"
                    b["last_batch_count"] = 0
                    b["last_deleted_count"] = 0
                    b["last_error"] = err
            return

        api_key = os.getenv("DASHBOARD_API_KEY") or os.getenv("PROSPER_API_KEY")
        bearer = os.getenv("PROSPER_BEARER_TOKEN")
        # New: token auto-refresh. If site code + email + password are set,
        # the auth provider can self-refresh on 401/403 without operator action.
        auth_provider = self._get_prosper_auth(base_url=base_url, globals_cfg=globals_cfg)
        has_auth_provider = bool(auth_provider and auth_provider.is_configured())
        if not api_key and not bearer and not has_auth_provider:
            log.warning(
                "[TrailerVisionApp] Prosper upload skipped: configure either "
                "(DASHBOARD_API_KEY / PROSPER_API_KEY / PROSPER_BEARER_TOKEN) "
                "or the login triple (PROSPER_SITE_CODE + PROSPER_EMAIL + PROSPER_PASSWORD)."
            )
            err = "no API key, bearer token, or login credentials"
            with self._upload_status_lock:
                self.upload_status["last_run_at"] = datetime.utcnow().isoformat() + "Z"
                for key in ("yard", "gate"):
                    b = self.upload_status[key]
                    b["last_result"] = "skipped"
                    b["last_batch_count"] = 0
                    b["last_deleted_count"] = 0
                    b["last_error"] = err
            return

        device_raw = os.getenv("EDGE_DEVICE_ID", "")
        batch_size = min(200, max(1, int(os.getenv("EDGE_UPLOAD_BATCH_SIZE", "100"))))
        gate_cutoff = min(3600, max(0, int(os.getenv("EDGE_GATE_UPLOAD_CUTOFF_SECONDS", "5"))))
        yard_records = self.video_frame_db.get_all_records(
            limit=batch_size, offset=0, is_processed=True, camera_id=None
        )
        gate_records = self.video_frame_db.get_gatevision_fused_records_pending_upload(
            limit=batch_size,
            cutoff_seconds=gate_cutoff,
        )
        if not yard_records and not gate_records:
            with self._upload_status_lock:
                self.upload_status["last_run_at"] = datetime.utcnow().isoformat() + "Z"
                for key in ("yard", "gate"):
                    b = self.upload_status[key]
                    b["last_result"] = "skipped"
                    b["last_batch_count"] = 0
                    b["last_deleted_count"] = 0
                    b["last_error"] = None
            return

        gate_id_map = globals_cfg.get("prosper_gate_id_map") or {}
        if not isinstance(gate_id_map, dict):
            gate_id_map = {}

        camera_id_map = globals_cfg.get("prosper_camera_id_map") or {}
        if not isinstance(camera_id_map, dict):
            camera_id_map = {}

        with self._upload_status_lock:
            self.upload_status["is_uploading"] = True
            self.upload_status["last_run_at"] = datetime.utcnow().isoformat() + "Z"

        y_ok: List[int] = []
        y_err: List[str] = []
        g_ok: List[int] = []
        g_err: List[str] = []
        # Best-effort image uploads (collected for status reporting only —
        # never block the parent event upload's success or DB deletion).
        img_errs: List[str] = []

        try:
            succeeded_ids: List[int] = []

            if yard_records:
                y_ok, y_err = upload_processed_yard_records_prosper(
                    yard_records,
                    base_url=base_url,
                    site_id=site_id,
                    device_id_raw=device_raw,
                    api_key=api_key,
                    bearer_token=bearer,
                    auth_provider=auth_provider if has_auth_provider else None,
                    source_system="YardVision",
                    camera_id_map=camera_id_map,
                )
                succeeded_ids.extend(y_ok)

            if gate_records:
                g_ok, g_err = upload_gatevision_records_prosper(
                    gate_records,
                    base_url=base_url,
                    site_id=site_id,
                    device_id_raw=device_raw,
                    gate_id_map=gate_id_map,
                    api_key=api_key,
                    bearer_token=bearer,
                    auth_provider=auth_provider if has_auth_provider else None,
                    source_system="GateVision",
                    camera_id_map=camera_id_map,
                )
                succeeded_ids.extend(g_ok)

            # ─── Best-effort image metadata upload ───────────────────────────
            # For every parent event we just uploaded successfully, also POST a
            # row to /api/sites/{siteId}/images/trailers with a fileUrl that
            # Prosper can pull. Skipped silently when:
            #   • EDGE_PUBLIC_BASE_URL is unset (operator opt-in)
            #   • the SQLite row has no image_path
            #   • the image is outside out/crops/ (we refuse to expose those)
            # Failures here are NEVER fatal: the gate/yard row already shipped
            # and will still be deleted from SQLite as success.
            edge_public_base = (
                os.getenv("EDGE_PUBLIC_BASE_URL")
                or globals_cfg.get("edge_public_base_url")
                or ""
            ).strip()
            if edge_public_base and succeeded_ids:
                try:
                    from app.prosper_image_upload import upload_record_images
                    camera_id_map = globals_cfg.get("prosper_camera_id_map") or {}
                    if not isinstance(camera_id_map, dict):
                        camera_id_map = {}
                    image_entity_type = os.getenv("PROSPER_IMAGE_ENTITY_TYPE") or globals_cfg.get(
                        "prosper_image_entity_type"
                    ) or "Trailer"
                    yard_succ_rows = [r for r in yard_records if r.get("id") in set(y_ok)]
                    gate_succ_rows = [r for r in gate_records if r.get("id") in set(g_ok)]
                    if yard_succ_rows:
                        _, errs = upload_record_images(
                            yard_succ_rows,
                            base_url=base_url,
                            site_id=site_id,
                            device_id_raw=device_raw,
                            edge_public_base_url=edge_public_base,
                            camera_id_map=camera_id_map,
                            api_key=api_key,
                            bearer_token=bearer,
                            auth_provider=auth_provider if has_auth_provider else None,
                            entity_type=image_entity_type,
                            source_system="YardVision",
                        )
                        img_errs.extend(errs)
                    if gate_succ_rows:
                        _, errs = upload_record_images(
                            gate_succ_rows,
                            base_url=base_url,
                            site_id=site_id,
                            device_id_raw=device_raw,
                            edge_public_base_url=edge_public_base,
                            camera_id_map=camera_id_map,
                            api_key=api_key,
                            bearer_token=bearer,
                            auth_provider=auth_provider if has_auth_provider else None,
                            entity_type=image_entity_type,
                            source_system="GateVision",
                        )
                        img_errs.extend(errs)
                except Exception as e:
                    log.warning("[TrailerVisionApp] Image metadata upload skipped/failed: %s", e)

            errors = list(y_err) + list(g_err) + list(img_errs)
            combined = yard_records + gate_records
            if succeeded_ids:
                # IDs are no longer unique across yardvision_records and gatevision_records
                # (each table has its own AUTOINCREMENT). Partition succeeded IDs against
                # the records we asked the uploader about, then delete from each table.
                yard_id_set = {r.get("id") for r in yard_records}
                gate_id_set = {r.get("id") for r in gate_records}
                succ_set = set(succeeded_ids)
                yard_succ_ids = [i for i in succeeded_ids if i in yard_id_set]
                gate_succ_ids = [i for i in succeeded_ids if i in gate_id_set]
                yard_del = self.video_frame_db.delete_yardvision_by_ids(yard_succ_ids)
                gate_del = self.video_frame_db.delete_gatevision_by_ids(gate_succ_ids)
                deleted = yard_del + gate_del
                uploaded_rows = [r for r in combined if r.get("id") in succ_set]
                self._delete_uploaded_images_and_crop_folders(uploaded_rows)
                log.info(
                    "[TrailerVisionApp] Prosper: uploaded %s row(s) (yard=%s gate=%s), deleted %s from SQLite",
                    len(succeeded_ids),
                    yard_del,
                    gate_del,
                    deleted,
                )

                def _branch_update(
                    had: bool,
                    ok_ids: List[int],
                    errs: List[str],
                    n_deleted: int,
                ) -> Dict:
                    if not had:
                        return {
                            "last_result": "skipped",
                            "last_batch_count": 0,
                            "last_deleted_count": 0,
                            "last_error": None,
                            "last_response_status": None,
                            "last_response_body": None,
                        }
                    n_ok = len(ok_ids)
                    err_s = "; ".join(errs[:5]) if errs else None
                    if n_ok and not errs:
                        res = "success"
                    elif n_ok and errs:
                        res = "partial"
                    elif errs:
                        res = "failed"
                    else:
                        res = "skipped"
                    return {
                        "last_result": res,
                        "last_batch_count": n_ok,
                        "last_deleted_count": n_deleted,
                        "last_error": err_s,
                        "last_response_status": 200 if n_ok else None,
                        "last_response_body": (str(errs)[:500] if errs else None),
                    }

                y_snap = _branch_update(bool(yard_records), y_ok, y_err, yard_del)
                g_snap = _branch_update(bool(gate_records), g_ok, g_err, gate_del)

                with self._upload_status_lock:
                    for k, v in y_snap.items():
                        self.upload_status["yard"][k] = v
                    for k, v in g_snap.items():
                        self.upload_status["gate"][k] = v
                    if yard_del:
                        ytot = self.upload_status["yard"].get("total_uploaded", 0) + yard_del
                        self.upload_status["yard"]["total_uploaded"] = ytot
                    if gate_del:
                        gtot = self.upload_status["gate"].get("total_uploaded", 0) + gate_del
                        self.upload_status["gate"]["total_uploaded"] = gtot
            else:
                def _branch_fail(had: bool, errs: List[str]) -> Dict:
                    if not had:
                        return {
                            "last_result": "skipped",
                            "last_batch_count": 0,
                            "last_deleted_count": 0,
                            "last_error": None,
                            "last_response_status": None,
                            "last_response_body": None,
                        }
                    if errs:
                        return {
                            "last_result": "failed",
                            "last_batch_count": 0,
                            "last_deleted_count": 0,
                            "last_error": "; ".join(errs[:5]),
                            "last_response_status": None,
                            "last_response_body": str(errs)[:500],
                        }
                    return {
                        "last_result": "skipped",
                        "last_batch_count": 0,
                        "last_deleted_count": 0,
                        "last_error": None,
                        "last_response_status": None,
                        "last_response_body": None,
                    }

                y_snap = _branch_fail(bool(yard_records), y_err)
                g_snap = _branch_fail(bool(gate_records), g_err)
                with self._upload_status_lock:
                    for k, v in y_snap.items():
                        self.upload_status["yard"][k] = v
                    for k, v in g_snap.items():
                        self.upload_status["gate"][k] = v
                if errors:
                    log.warning("[TrailerVisionApp] Prosper upload errors: %s", errors[:5])
        except Exception as e:
            log.warning("[TrailerVisionApp] Prosper upload error: %s", e)
            err = str(e)

            def _exc_branch(had: bool) -> Dict:
                if not had:
                    return {
                        "last_result": "skipped",
                        "last_batch_count": 0,
                        "last_deleted_count": 0,
                        "last_error": None,
                        "last_response_status": None,
                        "last_response_body": None,
                    }
                return {
                    "last_result": "failed",
                    "last_batch_count": 0,
                    "last_deleted_count": 0,
                    "last_error": err,
                    "last_response_status": None,
                    "last_response_body": None,
                }

            y_exc = _exc_branch(bool(yard_records))
            g_exc = _exc_branch(bool(gate_records))
            with self._upload_status_lock:
                for k, v in y_exc.items():
                    self.upload_status["yard"][k] = v
                for k, v in g_exc.items():
                    self.upload_status["gate"][k] = v
        finally:
            with self._upload_status_lock:
                self.upload_status["is_uploading"] = False

    def initialize_assets(self):
        """
        Initialize all assets required for automated processing.
        This includes OCR model loading and processing queue setup.
        
        Returns:
            Dict with 'success', 'message', and 'assets_loaded' status
        """
        assets_loaded = {
            'detector': self.detector is not None,
            'ocr': self.ocr is not None,
            'video_processor': self.video_processor is not None,
            'processing_queue': self.processing_queue is not None
        }
        
        try:
            # 1. Check detector
            if not self.detector:
                return {
                    'success': False,
                    'message': 'Detector not available. Please check model files.',
                    'assets_loaded': assets_loaded
                }
            
            # 2. Load OCR for per-video pipeline (OCR runs after each video)
            if self.ocr is None:
                log.info(f"[TrailerVisionApp] Loading OCR for automated processing...")
                self._initialize_ocr()
                assets_loaded['ocr'] = self.ocr is not None
            if not self.ocr:
                return {
                    'success': False,
                    'message': 'OCR model failed to load. Please check OCR model files.',
                    'assets_loaded': assets_loaded
                }
            
            # 3. Ensure video processor is available
            if not self.video_processor:
                return {
                    'success': False,
                    'message': 'Video processor not available.',
                    'assets_loaded': assets_loaded
                }
            
            # 4. Initialize processing queue if not already initialized
            if self.processing_queue is None:
                log.info(f"[TrailerVisionApp] Initializing processing queue...")
                
                # Define callbacks for extensibility
                def on_video_complete(video_path, crops_dir, results):
                    """Called when video processing completes."""
                    log.info(f"[TrailerVisionApp] Video processing complete: {Path(video_path).name}")
                    log.info(f"  - Crops directory: {crops_dir}")
                    # Extensibility point: Add server upload or other processing here
                    if results:
                        self.upload_to_server(video_path, crops_dir, {'type': 'video_processing', 'results': results})
                
                def on_ocr_complete(video_path, crops_dir, ocr_results):
                    """Called when OCR processing completes. Only processed data is uploaded (via periodic upload thread)."""
                    log.info(f"[TrailerVisionApp] OCR processing complete: {Path(video_path).name}")
                    log.info(f"  - Processed {len(ocr_results)} crops with OCR")
                    
                    # Store results in database (as per diagram requirement)
                    if self.video_frame_db and ocr_results:
                        self._store_ocr_results_in_db(video_path, crops_dir, ocr_results)
                    
                    # Upload to AWS is done only for processed records (after data processor assigns spots)
                    # See upload_processed_records_and_delete() and the periodic upload thread.
                    
                    # Delete video, crops, and GPS log permanently after processing is done
                    self._delete_processed_video_assets(video_path, crops_dir)
                
                try:
                    self.processing_queue = ProcessingQueueManager(
                        video_processor=self.video_processor,
                        ocr=self.ocr,
                        preprocessor=self.preprocessor,
                        on_video_complete=on_video_complete,
                        on_ocr_complete=on_ocr_complete,
                        defer_ocr=False,
                        on_video_queue_drained=None
                    )
                    log.info(f"[TrailerVisionApp] Processing queue manager initialized (OCR per video)")
                    assets_loaded['processing_queue'] = True
                    self._sync_offline_gpu_lock_to_gate_pipeline()
                    
                    # Set up video recorder callback for auto-processing
                    def on_chunk_saved(video_path, gps_log_path):
                        """Called when a video chunk is saved."""
                        # Extract camera_id from video path
                        video_name = Path(video_path).stem
                        # Format: camera_id_timestamp_chunkXXXX
                        parts = video_name.split('_')
                        camera_id = parts[0] if parts else "unknown"
                        
                        log.info(f"[TrailerVisionApp] Chunk saved: {Path(video_path).name}")
                        log.info(f"  - Queueing for processing...")
                        
                        # Queue video processing (use app detection_mode so car/trailer matches config)
                        if self.processing_queue:
                            self.processing_queue.queue_video_processing(
                                video_path=video_path,
                                camera_id=camera_id,
                                gps_log_path=gps_log_path,
                                detect_every_n=5,
                                detection_mode=getattr(self, 'detection_mode', 'trailer')
                            )
                    
                    # Update video recorder with callback.
                    # In `gate` mode the GateVisionPipeline already handles live detect→crop→OCR
                    # per vehicle, so auto-queueing whole 45s chunks would duplicate work and
                    # produce ghost events. Skip the callback wiring there.
                    if self.pipeline_mode == "gate":
                        log.info(
                            "[TrailerVisionApp] Gate pipeline: chunk-saved auto-queue disabled "
                            "(live gate pipeline handles real-time detection+OCR per vehicle)."
                        )
                    else:
                        self.video_recorder.on_chunk_saved = on_chunk_saved
                        log.info(f"[TrailerVisionApp] Video recorder callback configured for auto-processing")

                    # Start runtime watcher for manually copied videos in recordings folder.
                    # Off in gate mode — backlog runs only when dashboard button is pressed.
                    if self.pipeline_mode == "gate":
                        log.info(
                            "[TrailerVisionApp] Gate pipeline: recordings watcher disabled. "
                            "Drop files into out/recordings and press 'Process Backlog' in the dashboard to run them."
                        )
                    else:
                        if os.getenv("ENABLE_RECORDINGS_WATCHER", "0").lower() in ("1", "true", "yes", "on"):
                            self._start_recordings_watcher()
                    
                except Exception as e:
                    log.error("[TrailerVisionApp] Failed to initialize processing queue: %s", e)
                    import traceback
                    traceback.print_exc()
                    return {
                        'success': False,
                        'message': f'Failed to initialize processing queue: {str(e)}',
                        'assets_loaded': assets_loaded
                    }
            
            # All assets loaded successfully
            return {
                'success': True,
                'message': 'All assets initialized successfully',
                'assets_loaded': assets_loaded
            }
            
        except Exception as e:
            log.error("[TrailerVisionApp] Failed to initialize assets: %s", e)
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'message': f'Failed to initialize assets: {str(e)}',
                'assets_loaded': assets_loaded
            }
    
    def _get_gps_reference(self, camera_id: str) -> Optional[Dict[str, float]]:
        """
        Get GPS reference point from GPS sensor only.
        
        Returns:
            Dict with 'lat' and 'lon' keys, or None if sensor unavailable or no fix
        """
        if not self.gps_sensor:
            return None
        try:
            current_gps = self.gps_sensor.get_current_gps()
            if current_gps and current_gps.get('lat') is not None and current_gps.get('lon') is not None:
                return {'lat': current_gps['lat'], 'lon': current_gps['lon']}
        except Exception:
            pass
        return None
    
    def _project_to_world(self, camera_id: str, x_img: float, y_img: float, return_gps: bool = True, frame: Optional[np.ndarray] = None) -> Optional[tuple]:
        """
        Project image coordinates to world coordinates.
        
        Priority: 3D Projection > Homography
        Note: BEV projection has been removed - GPS coordinates come from GPS sensor/log files.
        
        Args:
            camera_id: Camera identifier
            x_img: Image X coordinate
            y_img: Image Y coordinate
            return_gps: If True and GPS reference available, return GPS coordinates (lat, lon).
                       If False or no GPS reference, return local meters (x, y).
            frame: Optional frame image (not used, kept for API compatibility)
            
        Returns:
            Tuple of (lat, lon) if return_gps=True and GPS reference available,
            Tuple of (x_world, y_world) in meters otherwise,
            or None if no projector available
        """
        # Priority 1: Use 3D projection if available
        if camera_id in self.camera_3d_projectors:
            try:
                projector = self.camera_3d_projectors[camera_id]
                x_world, y_world = projector.pixel_to_world(
                    x_img, y_img,
                    plane_z=0.0,  # Project to ground plane
                    return_gps=return_gps
                )
                return (x_world, y_world)
            except Exception as e:
                log.warning("3D projection failed for %s: %s", camera_id, e)
                # Fall through - return None if no projection available
        
        # Note: BEV projection removed - GPS coordinates come from GPS sensor/log files
        # If 3D projection is not available, return None
        # GPS coordinates should be obtained from GPS log files during video processing
        return None
    
    def _cleanup_gpu_memory(self, force: bool = False):
        """
        Clean up GPU memory after OCR operations to prevent accumulation.
        This is critical when processing multiple frames in sequence.
        
        Args:
            force: If True, always cleanup. If False, cleanup only periodically.
        """
        if not TORCH_AVAILABLE or not self.is_olmocr:
            return
        
        # OPTIMIZATION: Only cleanup periodically to reduce overhead
        if not force:
            if not hasattr(self, '_ocr_call_count'):
                self._ocr_call_count = 0
            self._ocr_call_count += 1
            # Only cleanup every 5 OCR operations to reduce overhead
            if self._ocr_call_count % 5 != 0:
                return
        
        if torch.cuda.is_available():
            try:
                # Wait for all GPU operations to complete
                torch.cuda.synchronize()
                # Clear CUDA cache
                torch.cuda.empty_cache()
                # Force Python garbage collection (less frequently)
                if force or (hasattr(self, '_ocr_call_count') and self._ocr_call_count % 10 == 0):
                    gc.collect()
                # Clear cache again after GC
                torch.cuda.empty_cache()
                gc.collect()
                # Clear cache again after GC
                torch.cuda.empty_cache()
            except Exception:
                # Silently fail - don't interrupt frame processing
                pass
    
    def _initialize_ocr(self):
        """
        Initialize OCR model (lazy loading).
        This is called after YOLO video processing is complete to save memory.

        Thread-safe: startup backlog, /api/start-application, and debug endpoints may
        call this concurrently; only one load runs and others wait then skip if done.
        """
        if self.ocr is not None:
            log.info(f"[TrailerVisionApp] OCR already initialized, skipping...")
            return

        with self._ocr_init_lock:
            if self.ocr is not None:
                log.info("[TrailerVisionApp] OCR initialized by another thread, skipping...")
                return
            self._initialize_ocr_inner()

    def _initialize_ocr_inner(self):
        globals_cfg = self.config.get('globals', {})

        log.info(f"[TrailerVisionApp] Initializing OCR...")
        # Signal to the MJPEG streamer (and any other heavy-allocator code paths)
        # that we're loading a multi-GB VLM into unified memory. On Jetson Orin,
        # concurrent JPEG encoding from /stream/<camera> requests during this
        # window is enough to OOM-kill the process near the end of weight load.
        self.gpu_load_in_progress = True
        
        # Default: oLmOCR only (Qwen3-VL per OCR_MODEL). Set OCR_ALLOW_FALLBACK=1 to restore EasyOCR/TrOCR/Paddle.
        allow_ocr_fallback = os.getenv("OCR_ALLOW_FALLBACK", "1").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.ocr = None
        
        # Try oLmOCR first (best accuracy for vertical text, complex layouts, and multi-language support)
        # Gate-only mode runs live YOLO on gate_* cameras while this thread starts up; importing
        # olmocr_recognizer can take many seconds (transformers/torch). If we unload YOLO only after
        # that import, YOLO + import + weight load compete for unified memory and the process may be
        # OOM-killed. Yard-only with only gate_* cams skips live detection, which is why it looked fine.
        restore_detector = False
        saved_mode = self.detection_mode
        unload_for_olmocr = (
            os.getenv("OCR_UNLOAD_DETECTOR_BEFORE_LOAD", "1").lower()
            not in ("0", "false", "no", "off")
            and TORCH_AVAILABLE
            and torch.cuda.is_available()
        )
        if unload_for_olmocr and self.detector is not None:
            with self.detector_runtime_lock:
                restore_detector = True
                log.info(
                    "[TrailerVisionApp] Temporarily unloading live detector before oLmOCR import/load "
                    "(set OCR_UNLOAD_DETECTOR_BEFORE_LOAD=0 to disable)."
                )
                self._unload_detector()
        try:
            from app.ocr.olmocr_recognizer import OlmOCRRecognizer
            # Model / device: env overrides config (Jetson: use NVIDIA torch + matching CUDA)
            ocr_model = os.getenv(
                "OCR_MODEL",
                globals_cfg.get("ocr_model_name", "Qwen/Qwen3-VL-4B-Instruct"),
            )
            ocr_device = os.getenv("OCR_DEVICE", "").strip() or os.getenv("DEVICE", "").strip() or None
            ocr_use_gpu = os.getenv("OCR_USE_GPU", "1").lower() not in (
                "0",
                "false",
                "no",
                "off",
            )
            # Use Qwen3-VL-4B-Instruct (or OCR_MODEL) for best OCR; Qwen3-VL-4B-Instruct-FP8 uses less VRAM
            self.ocr = OlmOCRRecognizer(
                model_name=ocr_model,
                device=ocr_device,
                use_gpu=ocr_use_gpu,
                fast_preprocessing=globals_cfg.get('ocr_fast_preprocessing', False)  # Enable via config
            )
            self.is_olmocr = True  # Mark as oLmOCR for GPU memory cleanup

            # Lazy load OCR model weights on demand (avoid startup OOM)
            preload_ocr = os.getenv("PRELOAD_OCR", "0").lower() in ("1", "true", "yes")
            if preload_ocr and hasattr(self.ocr, '_load_model'):
                try:
                    log.info("[TrailerVisionApp] Pre-loading OCR model into GPU VRAM on startup...")
                    self.ocr._load_model()
                except Exception as load_err:
                    log.warning("[TrailerVisionApp] Pre-loading OCR model skipped (will load on demand): %s", load_err)
                    import gc
                    gc.collect()
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()

            log.info(
                "✓ Using oLmOCR (%s) - recommended for high accuracy, especially vertical text",
                ocr_model,
            )
        except ImportError:
            log.error(
                "oLmOCR not available (pip3 install transformers torch qwen-vl-utils). "
                "Install dependencies or set OCR_ALLOW_FALLBACK=1 for legacy OCR engines."
            )
        except Exception as e:
            log.error(
                "oLmOCR initialization failed: %s. "
                "On Jetson try OCR_MODEL=Qwen/Qwen3-VL-4B-Instruct-FP8, OCR_USE_DEVICE_MAP=0, "
                "or increase swap; set OCR_ALLOW_FALLBACK=1 only if you accept non–Qwen-VL OCR.",
                e,
            )
            import gc
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        finally:
            if restore_detector and self.detector is None:
                gcfg = self.config.get("globals", {}) or {}
                # Read detection_mode dynamically at reload time, not the stale pre-OCR snapshot.
                # This means if the user changed the mode in the UI while OCR was loading, we
                # respect their current selection rather than reverting to what we had at startup.
                current_mode = getattr(self, 'detection_mode', saved_mode)
                with self.detector_runtime_lock:
                    self.detector = build_live_detector(current_mode, gcfg)
                    if self.detector is None:
                        log.error(
                            "[TrailerVisionApp] Could not reload detector after oLmOCR init (mode=%s). "
                            "Live detection will be unavailable until restart or apply_live_detection_mode.",
                            current_mode,
                        )
                    else:
                        self._sync_detector_to_consumers()
                        log.info(
                            "[TrailerVisionApp] Live detector reloaded after oLmOCR init (mode=%s).",
                            current_mode,
                        )
        
        if self.ocr is None and allow_ocr_fallback:
            try:
                from app.ocr.easyocr_recognizer import EasyOCRRecognizer
                self.ocr = EasyOCRRecognizer(languages=['en'], gpu=True)
                self.is_olmocr = False
                log.info("✓ Using EasyOCR (OCR_ALLOW_FALLBACK=1)")
            except ImportError:
                log.info("EasyOCR not available (pip3 install easyocr), trying TrOCR...")
            except Exception as e:
                log.info("EasyOCR initialization failed: %s, trying TrOCR...", e)

        if self.ocr is None and allow_ocr_fallback:
            # Try TrOCR (transformer-based, good accuracy for printed text)
            # Check for full TrOCR model first, then encoder-only
            trocr_engine_paths = [
                "models/trocr_full.engine",  # Full encoder-decoder model (preferred)
                "models/trocr.engine",        # Encoder-only or full model
            ]
            
            trocr_loaded = False
            for trocr_path in trocr_engine_paths:
                if os.path.exists(trocr_path):
                    try:
                        from app.ocr.trocr_recognizer import TrOCRRecognizer
                        # Try to find tokenizer in standard locations
                        tokenizer_paths = [
                            "models/trocr_base_printed",
                            "models/trocr-base-printed",
                        ]
                        model_dir = None
                        for path in tokenizer_paths:
                            if os.path.exists(path):
                                model_dir = path
                                break
                        
                        self.ocr = TrOCRRecognizer(trocr_path, model_dir=model_dir)
                        log.info(f"✓ Using TrOCR model: {trocr_path} (fallback - oLmOCR/EasyOCR not available)")
                        trocr_loaded = True
                        break
                    except ImportError:
                        log.info("TrOCR not available (transformers not installed), trying other OCR models...")
                        break
                    except Exception as e:
                        log.info(f"TrOCR initialization failed ({trocr_path}): {e}, trying other OCR models...")
                        continue
        
        if self.ocr is None and allow_ocr_fallback:
            ocr_path = None
            alphabet_path = None
            input_size = (320, 48)
            
            # Try PaddleOCR English-only (best for English/number text)
            if os.path.exists("models/paddleocr_rec_english.engine") and os.path.exists("app/ocr/ppocr_keys_en.txt"):
                ocr_path = "models/paddleocr_rec_english.engine"
                alphabet_path = "app/ocr/ppocr_keys_en.txt"
                log.info("Using PaddleOCR English-only model (fallback)")
            # Fallback to PaddleOCR multilingual
            elif os.path.exists("models/paddleocr_rec.engine") and os.path.exists("app/ocr/ppocr_keys_v1.txt"):
                ocr_path = "models/paddleocr_rec.engine"
                alphabet_path = "app/ocr/ppocr_keys_v1.txt"
                log.info("Using PaddleOCR multilingual model (fallback)")
            # Fallback to legacy CRNN engine
            elif os.path.exists("models/ocr_crnn.engine") and os.path.exists("app/ocr/alphabet.txt"):
                ocr_path = "models/ocr_crnn.engine"
                alphabet_path = "app/ocr/alphabet.txt"
                input_size = None  # CRNN uses default size
                log.info("Using legacy CRNN model (fallback)")
            
            if ocr_path and alphabet_path:
                if "paddleocr" in ocr_path.lower():
                    self.ocr = PlateRecognizer(ocr_path, alphabet_path, input_size=input_size)
                else:
                    self.ocr = PlateRecognizer(ocr_path, alphabet_path)
                log.info(f"Loaded OCR engine: {ocr_path} with alphabet: {alphabet_path}")
        
        # Re-enable streaming etc. as soon as the load-time spike is done, whether
        # OCR succeeded or failed. (set in _initialize_ocr_inner above.)
        self.gpu_load_in_progress = False

        if self.ocr is None:
            log.warning("No OCR engine available.")
            log.info("  - oLmOCR: pip3 install transformers torch qwen-vl-utils; set OCR_MODEL if needed.")
            if allow_ocr_fallback:
                log.info("  - Fallbacks were enabled but none succeeded (EasyOCR / TrOCR / Paddle / CRNN).")
            else:
                log.info("  - Legacy engines skipped (default). Set OCR_ALLOW_FALLBACK=1 to enable them.")
        else:
            # Update video processor OCR if it exists
            if hasattr(self, 'video_processor') and self.video_processor is not None:
                self.video_processor.ocr = self.ocr
                # Update is_olmocr flag in video processor
                self.video_processor.is_olmocr = self.is_olmocr
                log.info(f"[TrailerVisionApp] Updated video processor with OCR")
            self._sync_ocr_to_gate_pipeline()

            # Disable external OCR preprocessing for deep-learning/VLM engines since they handle it internally/better
            ocr_type = str(type(self.ocr))
            if 'OlmOCRRecognizer' in ocr_type or 'EasyOCRRecognizer' in ocr_type:
                if self.preprocessor is not None:
                    self.preprocessor.enable_ocr_preprocessing = False
                    log.info("[TrailerVisionApp] Disabled external OCR preprocessing for deep-learning/VLM OCR engine")

    def _clear_detector_from_consumers(self) -> None:
        """Drop detector references on workers so the model can be freed (GateVision holds its own ref)."""
        if getattr(self, "video_processor", None) is not None:
            self.video_processor.detector = None
        gate = self._get_gate_pipeline()
        if gate is not None:
            gate.detector = None

    def _sync_detector_to_consumers(self) -> None:
        """After (re)loading `self.detector`, keep VideoProcessor and GateVision in sync."""
        if getattr(self, "video_processor", None) is not None:
            self.video_processor.detector = self.detector
        gate = self._get_gate_pipeline()
        if gate is not None:
            gate.detector = self.detector

    def _sync_ocr_to_gate_pipeline(self) -> None:
        """GateVisionPipeline stores ocr at construction; refresh when OCR loads lazily."""
        gate = self._get_gate_pipeline()
        if gate is not None:
            gate.ocr = self.ocr
            if self.ocr is not None:
                log.info("[TrailerVisionApp] Synced active OCR engine (%s) to GateVision pipeline", type(self.ocr).__name__)

    def _sync_offline_gpu_lock_to_gate_pipeline(self) -> None:
        """Avoid concurrent live GateVision + offline queue CUDA work (Jetson OOM)."""
        gate = self._get_gate_pipeline()
        pq = getattr(self, "processing_queue", None)
        if gate is not None:
            gate.offline_gpu_lock = pq.gpu_lock if pq is not None else None
    
    def _unload_detector(self, force=False):
        """
        Unload YOLO detector to free GPU memory.
        This is called after YOLO video processing is complete.
        """
        if self.detector is None:
            log.info(f"[TrailerVisionApp] Detector already unloaded, skipping...")
            return
        
        # Determine if we should keep the detector loaded
        keep_env = os.getenv("DETECTOR_KEEP_LOADED", "").strip().lower()
        if keep_env in ("1", "true", "yes", "on"):
            keep_loaded = True
        elif keep_env in ("0", "false", "no", "off"):
            keep_loaded = False
        else:
            # Auto mode: keep loaded if we have >= 15GB VRAM
            from app.ocr.torch_cuda_compat import cuda_total_memory_bytes
            mem = cuda_total_memory_bytes(0)
            keep_loaded = (mem is not None and mem >= 15.0 * (1024**3))

        if keep_loaded and not force:
            log.info("[TrailerVisionApp] Keeping YOLO detector resident in GPU VRAM (DETECTOR_KEEP_LOADED is set or memory is sufficient).")
            return
        
        log.info(f"[TrailerVisionApp] Unloading YOLO detector to free GPU memory...")
        self._clear_detector_from_consumers()
        
        # Clean up detector based on type
        detector_type = type(self.detector).__name__
        
        if detector_type == "YOLOv8Detector":
            # YOLOv8 uses PyTorch/Ultralytics
            try:
                if hasattr(self.detector, 'model'):
                    # Move model to CPU and delete
                    if hasattr(self.detector.model, 'to'):
                        self.detector.model = self.detector.model.to('cpu')
                    del self.detector.model
                del self.detector
            except Exception as e:
                log.warning("[TrailerVisionApp] Error unloading YOLOv8 detector: %s", e)
        elif detector_type == "TrtEngineYOLO":
            # TensorRT engine - delete the engine
            try:
                if hasattr(self.detector, 'engine'):
                    del self.detector.engine
                if hasattr(self.detector, 'context'):
                    del self.detector.context
                del self.detector
            except Exception as e:
                log.warning("[TrailerVisionApp] Error unloading TensorRT detector: %s", e)
        
        self.detector = None

        # Clean up GPU memory
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                log.warning("[TrailerVisionApp] Error cleaning GPU memory: %s", e)

        log.info(f"[TrailerVisionApp] YOLO detector unloaded successfully")

    def _unload_ocr(self, force=False) -> None:
        """
        Unload the OCR (Qwen-VL) model to free GPU memory.

        Symmetric to ``_unload_detector``. Used by the gate-test flow when
        swapping back to YOLO between repeated test runs so YOLO + Qwen-VL
        are never resident at the same time on memory-constrained Jetson.

        Also clears the OCR reference on the video processor and gate
        pipeline so they don't keep dangling pointers to the freed model.
        """
        if getattr(self, "ocr", None) is None:
            log.info("[TrailerVisionApp] OCR already unloaded, skipping...")
            return

        # Determine if we should keep the OCR model loaded
        keep_env = os.getenv("OCR_KEEP_LOADED", "").strip().lower()
        if keep_env in ("1", "true", "yes", "on"):
            keep_loaded = True
        elif keep_env in ("0", "false", "no", "off"):
            keep_loaded = False
        else:
            # Auto mode: keep loaded if we have >= 15GB VRAM or if in stacker mode (which doesn't use YOLO)
            from app.ocr.torch_cuda_compat import cuda_total_memory_bytes
            mem = cuda_total_memory_bytes(0)
            keep_loaded = (mem is not None and mem >= 15.0 * (1024**3)) or getattr(self, "detection_mode", None) == "stacker"

        if keep_loaded and not force:
            log.info("[TrailerVisionApp] Keeping OCR model resident in GPU VRAM (detection_mode=stacker or memory is sufficient).")
            return

        log.info("[TrailerVisionApp] Unloading OCR model to free GPU memory...")
        # Drop references on consumers first; do this before we delete fields
        # on self.ocr so a concurrent call sees a consistent (None) state.
        try:
            if getattr(self, "video_processor", None) is not None:
                self.video_processor.ocr = None
                self.video_processor.is_olmocr = False
        except Exception as e:
            log.warning("[TrailerVisionApp] Error clearing OCR on video_processor: %s", e)
        try:
            gate = self._get_gate_pipeline()
            if gate is not None:
                gate.ocr = None
        except Exception as e:
            log.warning("[TrailerVisionApp] Error clearing OCR on gate_pipeline: %s", e)
        try:
            pq = getattr(self, "processing_queue", None)
            if pq is not None and hasattr(pq, "ocr"):
                pq.ocr = None
        except Exception as e:
            log.warning("[TrailerVisionApp] Error clearing OCR on processing_queue: %s", e)

        # Try to release the underlying transformer/torch resources.
        # OlmOCRRecognizer holds .model and .processor PyTorch objects.
        #
        # IMPORTANT: do NOT call ``.to("cpu")`` here. On Jetson Orin (unified
        # CPU/GPU memory) PyTorch still treats CPU and CUDA as logically
        # separate devices, so ``.to("cpu")`` allocates fresh CPU-side tensors
        # and copies the entire model through them — peaking at 2× the model
        # size (~16 GB for Qwen-VL on an 8 GB Orin) and OOM-killing the
        # process mid-unload. Just drop the references and let
        # ``gc.collect()`` + ``torch.cuda.empty_cache()`` below release the
        # weights in place.
        try:
            ocr_obj = self.ocr
            for attr in ("model", "processor", "tokenizer"):
                if getattr(ocr_obj, attr, None) is None:
                    continue
                try:
                    delattr(ocr_obj, attr)
                except Exception:
                    pass
        except Exception as e:
            log.warning("[TrailerVisionApp] Error releasing OCR submodules: %s", e)

        try:
            del self.ocr
        except Exception:
            pass
        self.ocr = None
        self.is_olmocr = False

        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                log.warning("[TrailerVisionApp] Error cleaning GPU memory after OCR unload: %s", e)

        log.info("[TrailerVisionApp] OCR model unloaded successfully")

    def _ensure_detector_loaded(self) -> bool:
        """
        Make sure the live YOLO detector is resident in memory.

        Returns True if a detector is available after this call. If a previous
        gate-test run unloaded YOLO to free memory for OCR, this rebuilds it
        for the current run so video processing can proceed.

        Caller is expected to have already unloaded any heavy model that
        would compete for GPU memory (e.g. OCR) before invoking this.
        """
        if getattr(self, "detector", None) is not None:
            return True
        gcfg = self.config.get("globals", {}) or {}
        saved_mode = getattr(self, "detection_mode", None) or "car"
        with self.detector_runtime_lock:
            if self.detector is not None:
                return True
            log.info(
                "[TrailerVisionApp] Reloading YOLO detector (mode=%s) for gate-test video phase...",
                saved_mode,
            )
            new_det = build_live_detector(saved_mode, gcfg)
            if new_det is None:
                log.error(
                    "[TrailerVisionApp] _ensure_detector_loaded: build_live_detector returned None (mode=%s)",
                    saved_mode,
                )
                return False
            self.detector = new_det
            self.detection_mode = saved_mode
            self._sync_detector_to_consumers()
        return True

    def apply_live_detection_mode(self, mode: str) -> Dict:
        """
        Swap the shared live detector for car vs trailer (YardVision + GateVision).

        Thread-safe with camera worker threads.
        """
        mode = (mode or "trailer").strip().lower()
        if mode not in ("car", "trailer"):
            return {"success": False, "message": "detection_mode must be car or trailer"}
        globals_cfg = self.config.get("globals", {})
        with self.detector_runtime_lock:
            self._unload_detector()
            new_det = build_live_detector(mode, globals_cfg)
            if new_det is None:
                log.error("[TrailerVisionApp] apply_live_detection_mode: failed to build detector for %s", mode)
                return {"success": False, "message": f"Could not load detector for mode={mode}"}
            self.detector = new_det
            self.detection_mode = mode
            self._sync_detector_to_consumers()
        log.info("[TrailerVisionApp] Live detection mode applied: %s", mode)
        return {"success": True, "detection_mode": mode}

    def _process_frame_legacy(self, camera_id: str, frame: np.ndarray, frame_count: int) -> None:
        """
        Process a single frame for a camera.
        
        Args:
            camera_id: Camera identifier
            frame: BGR image frame
            frame_count: Current frame number
        """
        # Store latest frame for video streaming (thread-safe)
        with self.frame_lock:
            self.latest_frames[camera_id] = frame.copy()
        
        globals_cfg = self.config.get('globals', {})
        detect_every_n = globals_cfg.get('detect_every_n', 5)
        save_frames = globals_cfg.get('save_frames', False)
        
        tracker = self.trackers[camera_id]
        metrics = self.camera_metrics[camera_id]
        
        # Preprocess frame for YOLO if enabled
        processed_frame = frame
        if self.preprocessor and self.preprocessor.enable_yolo_preprocessing:
            processed_frame = self.preprocessor.preprocess_for_yolo(frame)
        
        # Run detector every N frames
        detections = []
        if self.detector and frame_count % detect_every_n == 0:
            with self.detector_runtime_lock:
                all_detections = self.detector.detect(processed_frame)
            target_cls = expected_detection_class(self.detector)
            for det in all_detections:
                if det.get("cls", -1) == target_cls:
                    detections.append(det)
                # Optional: Log filtered detections for debugging
                # elif frame_count % 100 == 0:
                #     log.info(f"[TrailerVisionApp] Filtered non-trailer detection: cls={det.get('cls', -1)}, conf={det.get('conf', 0.0):.2f}")
        
        # Update tracker (now only contains trailer detections)
        tracks = tracker.update(detections, frame)
        
        # Process each track (all should be trailers now)
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            x1, y1, x2, y2 = bbox
            
            track_cls = track.get("cls", -1)
            want_cls = expected_detection_class(self.detector)
            if track_cls != -1 and track_cls != want_cls:
                if frame_count % 50 == 0:
                    log.info(
                        "[TrailerVisionApp] Skipping OCR for track %s: cls=%s (want %s)",
                        track_id,
                        track_cls,
                        want_cls,
                    )
                continue
            
            # Refine bounding box to rear face (focus on back side only)
            # This prevents OCR from detecting text on the sides of trailers
            orig_width = x2 - x1
            orig_height = y2 - y1
            orig_aspect = orig_width / orig_height if orig_height > 0 else 1.0
            
            # If aspect ratio suggests side view (wide), extract center portion for rear face
            # For wide detections (side view), the rear face is typically in the center
            if orig_aspect > 1.5:  # Wide detection (side view)
                center_x = (x1 + x2) / 2.0
                rear_width_ratio = 0.65  # Use 65% of width for rear face (centered)
                rear_width = int(orig_width * rear_width_ratio)
                rear_x1 = int(center_x - rear_width / 2)
                rear_x2 = int(center_x + rear_width / 2)
                # Keep full height
                rear_y1 = y1
                rear_y2 = y2
                
                # Clip to frame bounds
                h, w = frame.shape[:2]
                rear_x1 = max(0, min(rear_x1, w - 1))
                rear_x2 = max(rear_x1 + 1, min(rear_x2, w - 1))
                rear_y1 = max(0, min(rear_y1, h - 1))
                rear_y2 = max(rear_y1 + 1, min(rear_y2, h - 1))
                
                # Use refined coordinates for OCR (rear face only)
                x1, y1, x2, y2 = rear_x1, rear_y1, rear_x2, rear_y2
            # For narrow/tall detections (front/back view), use original bbox (already focused)
            
            # Crop region for OCR (now focused on rear face)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            # OPTIMIZATION: Check OCR cache first
            cache_key = (camera_id, track_id)
            should_run_ocr = True
            text = ""
            conf_ocr = 0.0
            ocr_method = "cached"
            
            if cache_key in self.ocr_cache:
                cached_result = self.ocr_cache[cache_key]
                cache_age = frame_count - cached_result['last_updated']
                
                # Use cached result if:
                # 1. It has good confidence (>= min_confidence)
                # 2. It's not too old (within max_age frames)
                # 3. Or we have a valid text result
                if (cached_result['conf'] >= self.ocr_min_confidence and 
                    cache_age < self.ocr_cache_max_age) or cached_result['text']:
                    text = cached_result['text']
                    conf_ocr = cached_result['conf']
                    should_run_ocr = False
                # Re-run OCR if cache is stale or low confidence
                elif cache_age >= self.ocr_cache_max_age:
                    should_run_ocr = True
                # For existing tracks, only run OCR periodically
                elif frame_count % self.ocr_run_every_n_frames != 0:
                    should_run_ocr = False
                    text = cached_result['text']  # Use cached even if low confidence
                    conf_ocr = cached_result['conf']
            
            # OPTIMIZATION: Skip OCR on very small crops (likely false positives)
            crop_area = (x2 - x1) * (y2 - y1)
            min_crop_area = 1000  # Minimum pixels for OCR (e.g., 32x32 = 1024)
            if crop_area < min_crop_area:
                should_run_ocr = False
                if cache_key not in self.ocr_cache:
                    text = ""
                    conf_ocr = 0.0
            
            # Run OCR with preprocessing (only if needed)
            if self.ocr and should_run_ocr:
                # OPTIMIZATION: Resize large crops before OCR to speed up processing
                # Large images take much longer to process
                h_crop, w_crop = crop.shape[:2]
                max_dimension = 640  # Maximum dimension for OCR (balance speed vs accuracy)
                if max(h_crop, w_crop) > max_dimension:
                    scale = max_dimension / max(h_crop, w_crop)
                    new_w = int(w_crop * scale)
                    new_h = int(h_crop * scale)
                    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                if self.preprocessor and self.preprocessor.enable_ocr_preprocessing:
                    # Get multiple preprocessed versions
                    preprocessed_crops = self.preprocessor.preprocess_for_ocr(crop)
                    
                    # OPTIMIZATION: Try OCR on each preprocessed version with early exit
                    ocr_results = []
                    for prep in preprocessed_crops:
                        try:
                            result = self.ocr.recognize(prep['image'])
                            if result.get('text', '').strip():
                                ocr_results.append({
                                    'text': result.get('text', ''),
                                    'conf': result.get('conf', 0.0),
                                    'method': prep['method']
                                })
                                # OPTIMIZATION: Early exit if we get high confidence result
                                if result.get('conf', 0.0) >= 0.85:
                                    break
                        except Exception as e:
                            if frame_count % 50 == 0:  # Log occasionally
                                log.info(f"[TrailerVisionApp] OCR error with {prep['method']}: {e}")
                    
                    # Select best result
                    if ocr_results:
                        best_result = self.preprocessor.select_best_ocr_result(ocr_results)
                        text = best_result['text']
                        conf_ocr = best_result['conf']
                        ocr_method = best_result['method']
                    else:
                        # Fallback: try original crop if all preprocessing failed
                        try:
                            ocr_result = self.ocr.recognize(crop)
                            text = ocr_result.get('text', '')
                            conf_ocr = ocr_result.get('conf', 0.0)
                            ocr_method = 'original-fallback'
                        except Exception as e:
                            if frame_count % 50 == 0:
                                log.info(f"[TrailerVisionApp] OCR fallback failed: {e}")
                            text = ""
                            conf_ocr = 0.0
                            ocr_method = 'error'
                else:
                    # Original OCR without preprocessing
                    try:
                        ocr_result = self.ocr.recognize(crop)
                        text = ocr_result['text']
                        conf_ocr = ocr_result['conf']
                        ocr_method = 'original'
                    except Exception as e:
                        if frame_count % 50 == 0:
                            log.info(f"[TrailerVisionApp] OCR error: {e}")
                        text = ""
                        conf_ocr = 0.0
                        ocr_method = 'error'
                
                # OPTIMIZATION: Cache OCR result (only cleanup GPU memory once after all OCR attempts)
                self.ocr_cache[cache_key] = {
                    'text': text,
                    'conf': conf_ocr,
                    'frame': frame_count,
                    'last_updated': frame_count
                }
                
                # OPTIMIZATION: Cleanup old cache entries to prevent memory leaks
                if len(self.ocr_cache) > self.ocr_cache_max_size:
                    # Remove oldest entries (by last_updated)
                    sorted_cache = sorted(self.ocr_cache.items(), key=lambda x: x[1]['last_updated'])
                    entries_to_remove = len(self.ocr_cache) - self.ocr_cache_max_size
                    for key, _ in sorted_cache[:entries_to_remove]:
                        del self.ocr_cache[key]
                
                # Clean GPU memory after OCR (only once, not after each preprocessing attempt)
                self._cleanup_gpu_memory()
            
            # Calculate ground contact point for trailer using learned linear model
            from app.bbox_to_image_coords_advanced import calculate_image_coords_from_bbox_with_config
            
            bbox = [float(x1), float(y1), float(x2), float(y2)]
            
            # Priority: If 3D projector available, use its bbox_to_ground_coords method
            # which properly accounts for trailer elevation
            if camera_id in self.camera_3d_projectors:
                try:
                    x_world, y_world = self.camera_3d_projectors[camera_id].bbox_to_ground_coords(
                        bbox,
                        method="backside_projection",
                        trailer_height=2.6  # Typical trailer back height in meters
                    )
                    world_coords_meters = (float(x_world), float(y_world))
                    # Image coords for reference (bottom-center)
                    center_x = (x1 + x2) / 2.0
                    bottom_y = float(y2)
                    image_coords = [float(center_x), float(bottom_y)]
                except Exception as e:
                    log.info(f"[TrailerVisionApp] Error in 3D bbox projection: {e}")
                    # Fall through to standard method
                    ground_x, ground_y = calculate_image_coords_from_bbox_with_config(bbox)
                    image_coords = [float(ground_x), float(ground_y)]
                    world_coords_meters = self._project_to_world(camera_id, ground_x, ground_y, return_gps=False, frame=frame)
            else:
                # Standard method: calculate image coords then project
                ground_x, ground_y = calculate_image_coords_from_bbox_with_config(bbox)
                image_coords = [float(ground_x), float(ground_y)]
                
                # Get world coordinates in meters first (for spot resolution)
                # Note: frame should be available in the calling context
                world_coords_meters = self._project_to_world(camera_id, ground_x, ground_y, return_gps=False, frame=frame)
            
            # Resolve parking spot (uses meters, same as GeoJSON)
            spot = "unknown"
            method = "no-calibration"
            if world_coords_meters and self.spot_resolver:
                x_world, y_world = world_coords_meters
                spot_result = self.spot_resolver.resolve(x_world, y_world)
                spot = spot_result['spot']
                method = spot_result['method']
            
            # Convert to GPS coordinates for output (if GPS reference available)
            # Uses live GPS sensor if available, otherwise falls back to static calibration reference
            world_coords_gps = None
            if world_coords_meters:
                gps_ref = self._get_gps_reference(camera_id)
                if gps_ref:
                    from app.container_utils import meters_to_gps
                    x_meters, y_meters = world_coords_meters
                    lat, lon = meters_to_gps(x_meters, y_meters, gps_ref['lat'], gps_ref['lon'])
                    world_coords_gps = (lat, lon)
            
            # Use GPS coordinates if available, otherwise use meters
            world_coords_output = world_coords_gps if world_coords_gps else world_coords_meters
            
            # Create event
            event = {
                'ts_iso': datetime.utcnow().isoformat(),
                'camera_id': camera_id,
                'track_id': track_id,
                'bbox': bbox,
                'image_coords': image_coords,  # Calculated image coordinates (ground contact point)
                'text': text,
                'conf': conf_ocr,
                'ocr_method': ocr_method,  # Track which preprocessing method was used
                'x_world': world_coords_output[0] if world_coords_output else None,
                'y_world': world_coords_output[1] if world_coords_output else None,
                'lat': world_coords_gps[0] if world_coords_gps else None,  # GPS latitude
                'lon': world_coords_gps[1] if world_coords_gps else None,  # GPS longitude
                'spot': spot,
                'method': method
            }
            
            # Log to CSV - COMMENTED OUT: Data is now stored in database instead
            # self.csv_logger.log(event)
            
            # Publish to event buses
            self.publisher.publish(event)
            
            # POST to REST ingest API if enabled
            if self.ingest_enabled:
                try:
                    requests.post(
                        self.ingest_url,
                        json=event,
                        timeout=1.0
                    )
                except Exception as e:
                    log.info(f"Error posting to ingest API: {e}")
            
            # Update last publish time
            metrics['last_publish'] = datetime.utcnow()
        
        # Save screenshot if enabled
        if save_frames and len(tracks) > 0:
            # Save frame with first track
            track_id = tracks[0]['track_id'] if tracks else None
            self.media_rotator.save_frame(camera_id, frame, track_id)
        
        # Update metrics
        metrics['frames_processed'] += 1
        
        # Update FPS (simple EMA)
        # In production, use proper time-based FPS calculation
        if metrics['frames_processed'] == 1:
            metrics['fps_ema'] = 30.0  # Initial estimate
        else:
            alpha = 0.1
            metrics['fps_ema'] = alpha * 30.0 + (1 - alpha) * metrics['fps_ema']
        
        # Update metrics server
        queue_depth = self.publisher.get_queue_depth()
        self.metrics_server.update_camera_metrics(
            camera_id,
            metrics['fps_ema'],
            metrics['frames_processed'],
            metrics['last_publish'],
            queue_depth
        )

    def _process_frame(self, camera_id: str, frame: np.ndarray, frame_count: int) -> None:
        """
        Process a single frame using the active pluggable vision pipeline.
        Falls back to legacy YardVision logic if pipeline is unavailable.
        """
        if self.pipeline is not None:
            self.pipeline.process_frame(camera_id, frame, frame_count)
            return
        self._process_frame_legacy(camera_id, frame, frame_count)
    
    def _process_camera(self, camera: Dict):
        """
        Process a single camera stream.
        
        Args:
            camera: Camera configuration dict
        """
        camera_id = camera['id']
        rtsp_url = camera['rtsp_url']
        
        # Detect if the source is a local video file
        is_file_stream = False
        try:
            from pathlib import Path
            p = Path(rtsp_url)
            if p.exists() and p.suffix.lower() in ('.mov', '.mp4', '.avi', '.mkv'):
                is_file_stream = True
        except Exception:
            pass

        if is_file_stream:
            force_file_process = os.getenv("EDGE_PROCESS_FILE_STREAMS_LIVE", "").lower() in ("1", "true")
            if not force_file_process:
                log.info(
                    "[TrailerVisionApp] Skipping live AI pipeline for %s because it is a local file (%s). "
                    "Use the dashboard 'Run Test' button to run OCR on recordings. "
                    "Camera will run in display-only mode.",
                    camera_id,
                    rtsp_url,
                )
                self._process_camera_display_only(camera)
                return

        width = camera.get('width', 1920)
        height = camera.get('height', 1080)
        fps_cap = camera.get('fps_cap', 30)
        
        globals_cfg = self.config.get('globals', {})
        use_gstreamer = globals_cfg.get('use_gstreamer', False)
        
        # Initialize active camera threads tracker
        if not hasattr(self, '_camera_threads'):
            self._camera_threads = {}
        if not hasattr(self, '_camera_stop_events'):
            self._camera_stop_events = {}

        stop_evt = threading.Event()
        self._camera_stop_events[camera_id] = stop_evt

        while self.running and not stop_evt.is_set():
            log.info(f"Opening live stream for {camera_id}: {rtsp_url}")
            cap = open_stream(rtsp_url, width, height, fps_cap, use_gstreamer)
            
            if cap is None:
                log.info("Stream %s not available currently (retrying in 4s)...", camera_id)
                for _ in range(4):
                    if not self.running or stop_evt.is_set():
                        return
                    time.sleep(1)
                continue
            
            frame_count = 0
            try:
                for ret, frame in frame_generator(cap):
                    if not ret or frame is None or not self.running or stop_evt.is_set():
                        break
                    
                    with self.frame_lock:
                        self.latest_frames[camera_id] = frame.copy()

                    if hasattr(self, 'pipeline') and self.pipeline is not None:
                        self.pipeline.process_frame(camera_id, frame, frame_count)
                    
                    frame_count += 1
            except KeyboardInterrupt:
                log.info(f"Interrupted processing for {camera_id}")
                break
            except Exception as e:
                log.info(f"Error in stream {camera_id}: {e}")
            finally:
                cap.release()
                log.info(f"Stream closed for {camera_id}")
            
            if not self.running or stop_evt.is_set():
                break
            time.sleep(2)

    def connect_live_camera(self, camera_id: str, rtsp_url: str, role: Optional[str] = None, gate_id: Optional[str] = None, direction: Optional[str] = None) -> Dict:
        """
        Dynamically connect, restart, or configure a live camera stream with role and direction.
        """
        import threading
        if not hasattr(self, '_camera_stop_events'):
            self._camera_stop_events = {}
        if not hasattr(self, '_camera_threads'):
            self._camera_threads = {}

        # Signal existing worker for this camera to stop
        if camera_id in self._camera_stop_events:
            self._camera_stop_events[camera_id].set()
            time.sleep(0.3)

        # Update in config
        cams = self.config.get('cameras', [])
        found = False
        for c in cams:
            if c.get('id') == camera_id:
                c['rtsp_url'] = rtsp_url
                if role:
                    c['role'] = role
                if gate_id:
                    c['gate_id'] = gate_id
                if direction:
                    c['direction'] = direction
                found = True
                break
        if not found:
            cams.append({
                'id': camera_id,
                'rtsp_url': rtsp_url,
                'role': role or 'gate_in_front',
                'gate_id': gate_id or 'gate-in',
                'direction': direction or 'INBOUND',
                'width': 1280,
                'height': 720,
                'fps_cap': 25
            })
            self.config['cameras'] = cams

        # Update pipeline role mappings
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            if hasattr(self.pipeline, 'camera_roles'):
                self.pipeline.camera_roles[camera_id] = role or 'gate_in_front'
            if hasattr(self.pipeline, 'camera_gate_ids'):
                self.pipeline.camera_gate_ids[camera_id] = gate_id or 'gate-in'

        # Launch fresh camera worker thread
        cam_cfg = next((c for c in cams if c.get('id') == camera_id), {
            'id': camera_id, 'rtsp_url': rtsp_url, 'width': 1280, 'height': 720, 'fps_cap': 25
        })

        t = threading.Thread(
            target=self._process_camera,
            args=(cam_cfg,),
            daemon=True,
            name=f"CamWorker-{camera_id}"
        )
        t.start()
        self._camera_threads[camera_id] = t

        log.info("[TrailerVisionApp] Started live camera worker for %s -> %s (role=%s, dir=%s)", camera_id, rtsp_url, role, direction)
        return {'success': True, 'camera_id': camera_id, 'rtsp_url': rtsp_url, 'role': role, 'direction': direction}
    
    def run(self):
        """Run the main application loop."""
        log.info("Starting Trailer Vision Edge application...")
        self.running = True

        run_full_processing = True
        mode = getattr(self.pipeline, "vision_mode", "hybrid")
        if mode == "hybrid":
            log.info(
                "Vision pipeline hybrid: YardVision + GateVision — cameras routed by role (gate_* → GateVision)."
            )
        elif mode == "yard":
            log.info("Vision pipeline yard-only: YardVision for non-gate cameras; gate_* cameras skip vision.")
        else:
            log.info("Vision pipeline gate-only: GateVision for gate_* cameras; other roles skip vision.")

        # Process each camera in a separate thread
        import threading

        threads = []
        for camera in self.config.get('cameras', []):
            target = self._process_camera if run_full_processing else self._process_camera_display_only
            thread = threading.Thread(
                target=target,
                args=(camera,),
                daemon=True
            )
            thread.start()
            threads.append(thread)
        
        # Keep application running for video processing and camera display
        try:
            for thread in threads:
                thread.join()
            # If streams are finished (e.g. video file ended), keep the app alive so the user can inspect the UI
            if self.running:
                log.info("All video streams finished. Keeping server alive for UI/metrics access. Press Ctrl+C to exit.")
                while self.running:
                    time.sleep(1)
        except KeyboardInterrupt:
            log.info("\nShutting down...")
            self.stop()
    
    def _process_camera_display_only(self, camera: Dict):
        """
        Process camera feed for display only (no full processing).
        This just captures frames and makes them available for streaming.
        
        Args:
            camera: Camera configuration dict
        """
        camera_id = camera['id']
        rtsp_url = camera['rtsp_url']
        width = camera.get('width', 1920)
        height = camera.get('height', 1080)
        fps_cap = camera.get('fps_cap', 30)
        
        globals_cfg = self.config.get('globals', {})
        use_gstreamer = globals_cfg.get('use_gstreamer', False)
        
        log.info(f"Opening camera stream for display: {camera_id}: {rtsp_url}")
        cap = open_stream(rtsp_url, width, height, fps_cap, use_gstreamer)
        
        if cap is None:
            log.warning("Failed to open stream for %s", camera_id)
            # Still register camera in metrics so dashboard shows it (even if stream failed)
            if self.metrics_server:
                self.metrics_server.update_camera_metrics(
                    camera_id,
                    fps_ema=0.0,
                    frames_processed_count=0,
                    last_publish=datetime.utcnow(),
                    queue_depth=0
                )
            return
        
        # Store the camera capture for streaming
        if not hasattr(self, 'camera_captures'):
            self.camera_captures = {}
        self.camera_captures[camera_id] = cap
        
        frame_count = 0
        
        # Register camera in metrics immediately so dashboard shows it
        if self.metrics_server:
            self.metrics_server.update_camera_metrics(
                camera_id,
                fps_ema=0.0,  # Will update when frames start coming
                frames_processed_count=0,
                last_publish=datetime.utcnow(),
                queue_depth=0
            )
        
        try:
            for ret, frame in frame_generator(cap):
                if not ret or frame is None:
                    break
                
                if not self.running:
                    break
                
                # Store latest frame for streaming (no processing)
                with self.frame_lock:
                    self.latest_frames[camera_id] = frame.copy()
                
                # Write frame to video recorder if recording
                if self.video_recorder and self.video_recorder.is_recording():
                    self.video_recorder.write_frame(frame)
                
                frame_count += 1
                
                # Update camera metrics periodically for dashboard display (every 30 frames = ~1 second)
                if frame_count % 30 == 0 and self.metrics_server:
                    self.metrics_server.update_camera_metrics(
                        camera_id,
                        fps_ema=30.0,  # Display only, estimate 30 FPS
                        frames_processed_count=frame_count,
                        last_publish=datetime.utcnow(),
                        queue_depth=0
                    )
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.033)  # ~30 FPS
                
        except KeyboardInterrupt:
            log.info(f"Interrupted display stream for {camera_id}")
        except Exception as e:
            log.info(f"Error displaying {camera_id}: {e}")
        finally:
            if camera_id in self.camera_captures:
                del self.camera_captures[camera_id]
            cap.release()
            log.info(f"Closed display stream for {camera_id}")

    def get_latest_frame(self, camera_id: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Get the most recent video frame for a given camera or any available camera.
        """
        with self.frame_lock:
            if camera_id:
                if camera_id in self.latest_frames:
                    f = self.latest_frames[camera_id]
                    return f.copy() if f is not None else None
                return None
            elif self.latest_frames:
                for cid, f in self.latest_frames.items():
                    if f is not None:
                        return f.copy()
        return None
    
    def start_recording(self, camera_id: str = None) -> Dict:
        """
        Start video recording with GPS logging.
        
        Args:
            camera_id: Camera ID to record (uses first camera if None)
            
        Returns:
            Dict with 'success', 'message', and optional 'video_path', 'gps_log_path'
        """
        if not self.video_recorder:
            return {
                'success': False,
                'message': 'Video recorder not initialized'
            }
        
        if self.video_recorder.is_recording():
            return {
                'success': False,
                'message': 'Already recording. Stop current recording first.'
            }
        
        # Get camera config
        cameras = self.config.get('cameras', [])
        if not cameras:
            return {
                'success': False,
                'message': 'No cameras configured'
            }
        
        # Use specified camera or first camera
        camera = None
        if camera_id:
            camera = next((c for c in cameras if c['id'] == camera_id), None)
            if not camera:
                return {
                    'success': False,
                    'message': f'Camera {camera_id} not found'
                }
        else:
            camera = cameras[0]
        
        camera_id = camera['id']
        width = camera.get('width', 1920)
        height = camera.get('height', 1080)
        fps = camera.get('fps_cap', 30)
        
        # Reset app state so we are no longer "gracefully shutting down" (new recording session)
        with self.shutdown_lock:
            self.recording_stopped = False
        # Notify queue that recording started (reset deferred-OCR state for this session)
        if self.processing_queue and getattr(self.processing_queue, 'notify_recording_started', None):
            self.processing_queue.notify_recording_started()
        
        # Start recording
        success = self.video_recorder.start_recording(camera_id, width, height, fps)
        
        if success:
            return {
                'success': True,
                'message': f'Recording started for camera {camera_id}',
                'camera_id': camera_id
            }
        else:
            return {
                'success': False,
                'message': 'Failed to start recording'
            }
    
    def stop_recording(self) -> Dict:
        """
        Stop video recording but continue processing remaining videos.
        Recording stops immediately, but video processing and OCR continue in background.
        
        Returns:
            Dict with 'success', 'message', 'video_path', 'gps_log_path', 'processing_ongoing'
        """
        if not self.video_recorder:
            return {
                'success': False,
                'message': 'Video recorder not initialized'
            }
        
        with self.shutdown_lock:
            if self.recording_stopped:
                # Already stopped, check if processing is still ongoing
                is_processing = self.is_processing_ongoing()
                return {
                    'success': True,
                    'message': 'Recording already stopped',
                    'processing_ongoing': is_processing
                }
        
        if not self.video_recorder.is_recording():
            return {
                'success': False,
                'message': 'Not currently recording'
            }
        
        # Stop recording
        video_path, gps_log_path = self.video_recorder.stop_recording()
        
        # Mark recording as stopped (but processing continues)
        with self.shutdown_lock:
            self.recording_stopped = True
        
        # Notify queue so deferred OCR can run when video queue drains
        if self.processing_queue and getattr(self.processing_queue, 'notify_recording_stopped', None):
            self.processing_queue.notify_recording_stopped()
        
        # Check if there are remaining videos to process
        is_processing = self.is_processing_ongoing()
        processing_status = self.get_processing_status()
        
        message = 'Recording stopped'
        if is_processing:
            video_queue = processing_status.get('video_queue_size', 0)
            ocr_queue = processing_status.get('ocr_queue_size', 0)
            message += f'. Processing {video_queue} video(s) and {ocr_queue} OCR job(s) in background'
        
        return {
            'success': True,
            'message': message,
            'video_path': video_path,
            'gps_log_path': gps_log_path,
            'processing_ongoing': is_processing,
            'processing_status': processing_status
        }
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.video_recorder.is_recording() if self.video_recorder else False
    
    def is_processing_ongoing(self) -> bool:
        """Check if video processing or OCR is still ongoing."""
        if not self.processing_queue:
            return False
        
        status = self.processing_queue.get_status()
        return (status.get('video_queue_size', 0) > 0 or 
                status.get('ocr_queue_size', 0) > 0 or
                status.get('processing_video', False) or
                status.get('processing_ocr', False))
    
    def is_gracefully_shutting_down(self) -> bool:
        """Check if recording is stopped but processing is still ongoing."""
        with self.shutdown_lock:
            return self.recording_stopped and self.is_processing_ongoing()
    
    def get_processing_status(self) -> Dict:
        """
        Get processing queue status.
        
        Returns:
            Dict with processing queue status information
        """
        if self.processing_queue:
            return self.processing_queue.get_status()
        return {
            'processing_video': False,
            'processing_ocr': False,
            'video_queue_size': 0,
            'ocr_queue_size': 0,
            'stats': {
                'videos_queued': 0,
                'videos_processed': 0,
                'ocr_jobs_queued': 0,
                'ocr_jobs_processed': 0,
                'errors': 0
            }
        }
    
    def _run_deferred_ocr(self, pending_jobs: List[Dict]):
        """
        Run OCR once on all pending crop directories (called when recording stopped and video queue drained).
        Loads OCR on first use, then processes each job and stores/uploads results.
        """
        if not pending_jobs:
            log.info("[TrailerVisionApp] Deferred OCR: no pending jobs")
            return
        log.info(f"[TrailerVisionApp] Deferred OCR: loading OCR and processing {len(pending_jobs)} crop directory(ies)")
        # Load OCR if not already loaded (only now to avoid GPU use during capture/video processing)
        if self.ocr is None:
            self._initialize_ocr()
        if not self.ocr:
            log.error("[TrailerVisionApp] Deferred OCR: failed to load OCR, skipping")
            return
        from app.batch_ocr_processor import BatchOCRProcessor
        for i, job in enumerate(pending_jobs):
            video_path = job.get('video_path', '')
            crops_dir = job.get('crops_dir', '')
            camera_id = job.get('camera_id', '')
            if not crops_dir or not Path(crops_dir).exists():
                log.warning(f"[TrailerVisionApp] Deferred OCR: skip missing crops_dir {crops_dir}")
                continue
            try:
                batch_processor = BatchOCRProcessor(self.ocr, self.preprocessor)
                ocr_results = batch_processor.process_crops_directory(crops_dir)
                combined_results = batch_processor.match_ocr_to_detections(crops_dir, ocr_results)
                if self.video_frame_db and combined_results:
                    self._store_ocr_results_in_db(video_path, crops_dir, combined_results)
                # Upload to AWS is only for processed records (periodic upload thread)
                log.info(f"[TrailerVisionApp] Deferred OCR: completed {i+1}/{len(pending_jobs)} {Path(crops_dir).name}")
            except Exception as e:
                log.exception(f"[TrailerVisionApp] Deferred OCR error for {crops_dir}: {e}")
            finally:
                self._cleanup_gpu_memory()
        log.info("[TrailerVisionApp] Deferred OCR: all jobs completed")
    
    def _store_ocr_results_in_db(
        self,
        video_path: str,
        crops_dir: str,
        ocr_results: List[Dict],
        *,
        video_path_db: Optional[str] = None,
        allow_missing_gps: bool = False,
    ) -> None:
        """
        Store OCR results in database (as per diagram requirement).
        
        Args:
            video_path: Path to source video (filesystem)
            crops_dir: Directory containing crops
            ocr_results: List of combined OCR results with GPS data
            video_path_db: Optional value stored in DB video_path column (e.g. gatevision:test:...)
            allow_missing_gps: If True, insert rows when lat/lon are absent (offline / gate tests)
        """
        if not self.video_frame_db:
            return
            
        # Check env variable ALLOW_MISSING_GPS as fallback
        env_allow_missing = os.getenv("ALLOW_MISSING_GPS", "0") == "1"
        effective_allow_missing = allow_missing_gps or env_allow_missing
        
        stored_count = 0
        skipped_invalid = 0
        skipped_lowconf = 0
        # Optional minimum OCR confidence to store a row. Defaults to 0.0 to
        # preserve prior behavior; raise via env to drop low-confidence noise
        # (e.g. ``OCR_STORE_MIN_CONFIDENCE=0.3``). 0.0 = no threshold.
        try:
            min_store_conf = float(os.getenv("OCR_STORE_MIN_CONFIDENCE", "0") or 0)
        except (TypeError, ValueError):
            min_store_conf = 0.0

        # ── Per-track deduplication ──────────────────────────────────────────
        # Multiple crops per physical trailer (up to 3) are OCR'd for coverage,
        # but we only want to store ONE record per track in the DB to avoid
        # sending the same trailer to the API multiple times.
        #
        # Strategy: group results by track_id, pick the best one per track:
        #   1. Prefer results that contain digits (trailer IDs) over text-only
        #   2. Among those with digits, prefer longer text (more complete ID)
        #   3. Among equal-length, prefer higher confidence
        from collections import defaultdict
        track_groups = defaultdict(list)
        ungrouped = []
        for result in ocr_results:
            tid = result.get('track_id')
            if tid is not None:
                track_groups[tid].append(result)
            else:
                ungrouped.append(result)

        deduped_results = list(ungrouped)  # keep ungrouped results as-is
        for tid, group in track_groups.items():
            # Score each result in the group
            def _dedup_score(r):
                text = (r.get('ocr_text') or '').strip()
                conf = float(r.get('ocr_conf') or 0.0)
                has_digits = any(c.isdigit() for c in text)
                # Extract just the trailer-number portion for length scoring
                try:
                    from app.container_utils import _extract_trailer_and_scac
                    trailer_part, _ = _extract_trailer_and_scac(text)
                    trailer_clean = trailer_part or ""
                except Exception:
                    trailer_clean = text
                digit_count = sum(1 for c in trailer_clean if c.isdigit())
                # Primary: has digits at all (100 pts)
                # Secondary: number of digits in trailer portion (0-10 pts each)
                # Tertiary: confidence (0-1)
                return (100 if has_digits else 0) + digit_count * 10 + conf

            best = max(group, key=_dedup_score)
            deduped_results.append(best)
            if len(group) > 1:
                log.info(
                    "[TrailerVisionApp] Dedup track %s: kept '%s' (conf=%.2f) from %d crops",
                    tid,
                    (best.get('ocr_text') or '')[:40],
                    float(best.get('ocr_conf') or 0),
                    len(group),
                )

        log.info(
            "[TrailerVisionApp] Deduplication: %d crops → %d unique records",
            len(ocr_results), len(deduped_results),
        )
        # ── End deduplication ────────────────────────────────────────────────

        # ── Global trailer ID deduplication across tracks ────────────────────
        # To prevent the same trailer from being recorded/uploaded twice from 
        # the same video run (e.g. '322099' and 'IBHU322099'), we perform a 
        # global deduplication matching subsets/substrings.
        final_results = []
        scored_results = []
        for r in deduped_results:
            raw_plate = (r.get('ocr_text') or '').strip()
            if not raw_plate:
                continue
            
            # Clean and parse the trailer ID
            try:
                from app.container_utils import _extract_trailer_and_scac, _clean_trailer_number
                trailer_part, scac_part = _extract_trailer_and_scac(raw_plate)
                cleaned_num = _clean_trailer_number(trailer_part) or ""
                cleaned_scac = (scac_part or "").strip().upper()
            except Exception:
                cleaned_num = raw_plate
                cleaned_scac = ""
                
            full_code = f"{cleaned_scac}{cleaned_num}" if cleaned_scac else cleaned_num
            # Score: prefer longer codes (which include SCAC prefix) and higher confidence
            score = len(full_code) * 10 + float(r.get('ocr_conf') or 0.0)
            scored_results.append({
                'result': r,
                'num': cleaned_num,
                'scac': cleaned_scac,
                'full_code': full_code,
                'score': score
            })

        # Process from highest score to lowest score (longest/most confident first)
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        seen_numbers = set()
        
        for item in scored_results:
            num = item['num']
            full_code = item['full_code']
            if not num or num == "UNKNOWN":
                final_results.append(item['result'])
                continue
                
            # Check if this number is a subset of/or matches an already processed trailer ID
            is_dup = False
            for existing in seen_numbers:
                if num in existing or existing in num:
                    is_dup = True
                    log.info(
                        "[TrailerVisionApp] Global deduplication: skipping '%s' (conf=%.2f) "
                        "because it matches existing '%s'",
                        full_code,
                        float(item['result'].get('ocr_conf') or 0),
                        existing
                    )
                    break
            
            if not is_dup:
                seen_numbers.add(full_code)
                seen_numbers.add(num)
                final_results.append(item['result'])
                
        deduped_results = final_results
        log.info(
            "[TrailerVisionApp] Global Deduplication: %d records post-track-dedup → %d unique final records",
            len(scored_results), len(deduped_results)
        )

        for result in deduped_results:
            try:
                # Extract data from combined result
                raw_plate = result.get('ocr_text', '').strip()
                if not raw_plate:
                    skipped_invalid += 1
                    continue  # Skip records without OCR text

                # ── Extract trailer ID from VLM output before sanitizing ─────
                # The VLM multi-pass output can be very long (80+ chars), e.g.:
                #   "Carrier R53275 TRANSPORT LOWELL WIS. CARTIER TRANSFERS ..."
                # Sanitize rejects text > MAX_PLATE_LENGTH (50). So we extract
                # the actual trailer number first using the same scoring logic
                # the upload path uses, then sanitize THAT.
                try:
                    from app.container_utils import _extract_trailer_and_scac, _clean_trailer_number
                    trailer_part, scac_part = _extract_trailer_and_scac(raw_plate)
                    cleaned = _clean_trailer_number(trailer_part)
                    if cleaned and cleaned != "UNKNOWN":
                        scac_clean = (scac_part or "").strip().upper()
                        if scac_clean and not cleaned.startswith(scac_clean):
                            licence_plate = f"{scac_clean}{cleaned}"
                        else:
                            licence_plate = cleaned
                    else:
                        # Fallback: try sanitizing the raw text directly
                        from app.ocr.plate_validation import sanitize_plate_for_storage
                        licence_plate = sanitize_plate_for_storage(raw_plate)
                except ImportError:
                    from app.ocr.plate_validation import sanitize_plate_for_storage
                    licence_plate = sanitize_plate_for_storage(raw_plate)
                # ── End extraction ───────────────────────────────────────────

                if not licence_plate:
                    skipped_invalid += 1
                    log.info(
                        "[TrailerVisionApp] OCR storage: rejecting invalid/placeholder text %r "
                        "(crop=%s, conf=%s)",
                        raw_plate,
                        result.get('crop_path') or result.get('crop_filename'),
                        result.get('ocr_conf'),
                    )
                    continue  # Skip: invalid / placeholder / prompt-leak text
                # Enforce optional minimum confidence so the operator can drop
                # noisy low-confidence rows by raising OCR_STORE_MIN_CONFIDENCE.
                try:
                    row_conf = float(result.get('ocr_conf') or 0.0)
                except (TypeError, ValueError):
                    row_conf = 0.0
                if min_store_conf > 0.0 and row_conf < min_store_conf:
                    skipped_lowconf += 1
                    log.info(
                        "[TrailerVisionApp] OCR storage: skipping low-confidence row "
                        "%r conf=%.3f < min=%.3f (crop=%s)",
                        licence_plate,
                        row_conf,
                        min_store_conf,
                        result.get('crop_path') or result.get('crop_filename'),
                    )
                    continue
                
                # Get GPS coordinates: combined results have 'lat'/'lon' from crop metadata (from GPS log during video processing)
                latitude = result.get('lat')
                longitude = result.get('lon')
                if latitude is None or longitude is None:
                    world_coords = result.get('world_coords')
                    if isinstance(world_coords, (list, tuple)) and len(world_coords) >= 2:
                        latitude, longitude = world_coords[0], world_coords[1]
                    elif isinstance(world_coords, dict):
                        latitude = world_coords.get('lat') or world_coords.get('x_world')
                        longitude = world_coords.get('lon') or world_coords.get('y_world')
                
                # Get other fields
                timestamp_str = result.get('timestamp', '')
                confidence = result.get('ocr_conf', 0.0)
                image_path = result.get('crop_path', '')
                camera_id = result.get('camera_id', '')
                frame_number = result.get('frame_count', 0)
                track_id = result.get('track_id')
                
                # Speed and barrier from GPS (if available in result)
                speed = result.get('speed')
                barrier = result.get('barrier') or result.get('heading')
                
                # Convert timestamp
                timestamp = None
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except:
                        pass
                
                store_vp = video_path_db if video_path_db else video_path
                cid_low = (camera_id or "").lower()
                if not video_path_db and (cid_low.startswith("gate") or "gate_" in cid_low):
                    store_vp = f"gatevision:{camera_id}:gate_pass"
                gps_ok = latitude is not None and longitude is not None
                if gps_ok or effective_allow_missing:
                    lat_v = float(latitude) if latitude is not None else None
                    lon_v = float(longitude) if longitude is not None else None
                    self.video_frame_db.insert_frame_record(
                        licence_plate_trailer=licence_plate,
                        latitude=lat_v,
                        longitude=lon_v,
                        speed=speed,
                        barrier=barrier,
                        confidence=confidence,
                        image_path=image_path,
                        camera_id=camera_id,
                        video_path=store_vp,
                        frame_number=frame_number,
                        track_id=track_id,
                        timestamp=timestamp
                    )
                    stored_count += 1
            except Exception as e:
                log.info(f"[TrailerVisionApp] Error storing OCR result in database: {e}")

        # Single summary line so the operator can see at a glance how many
        # OCR results were dropped vs stored. ``skipped_invalid`` covers empty
        # text + placeholder values (UNKNOWN/NONE/N/A/etc.) + too-short noise
        # + too-long text + prompt leaks. ``skipped_lowconf`` only fires when
        # OCR_STORE_MIN_CONFIDENCE is set above 0.
        log.info(
            "[TrailerVisionApp] Stored %d records in database "
            "(skipped: %d invalid/placeholder, %d below min-confidence)",
            stored_count, skipped_invalid, skipped_lowconf,
        )
    
    def _delete_uploaded_images_and_crop_folders(self, records: List[Dict]):
        """
        After successful upload, delete local image files and empty crop folders for the uploaded records.

        Skips GateVision recordings test rows (gatevision:test-*:gate_pass) so evidence crops stay on disk.
        """
        from app.prosper_gate_upload import is_gatevision_test_recording_row

        deleted_files = 0
        dirs_to_check = set()
        for r in records:
            if is_gatevision_test_recording_row(r):
                continue
            img_path = r.get("image_path")
            if not img_path:
                continue
            p = Path(img_path)
            if p.is_file():
                try:
                    p.unlink()
                    deleted_files += 1
                except Exception as e:
                    log.warning("[TrailerVisionApp] Failed to delete image %s: %s", img_path, e)
            if p.parent and p.parent != p:
                dirs_to_check.add(p.parent)
        # Remove empty crop directories (and parents up to out/ or cwd)
        try:
            stop_at = Path("out").resolve() if Path("out").exists() else Path.cwd()
        except Exception:
            stop_at = Path.cwd()
        for dir_entry in sorted(dirs_to_check, key=lambda x: len(x.parts), reverse=True):
            try:
                current = Path(dir_entry).resolve()
                while current.exists() and current.is_dir() and current != stop_at:
                    if any(current.iterdir()):
                        break
                    current.rmdir()
                    log.debug("[TrailerVisionApp] Removed empty crop dir: %s", current)
                    current = current.parent
            except Exception as e:
                log.debug("[TrailerVisionApp] Skip removing dir %s: %s", dir_entry, e)
        if deleted_files or dirs_to_check:
            log.info("[TrailerVisionApp] Deleted %s image file(s) and cleaned empty crop folder(s)", deleted_files)

    def _delete_processed_video_assets(self, video_path: str, crops_dir: str):
        """
        Permanently delete video file and GPS log after processing is complete.
        Crops folder is kept. Called from on_ocr_complete once DB and upload are done.
        """
        video_p = Path(video_path)
        # 1. Delete video file
        if video_p.is_file():
            try:
                video_p.unlink()
                log.info(f"[TrailerVisionApp] Deleted video: {video_p.name}")
            except Exception as e:
                log.warning(f"[TrailerVisionApp] Failed to delete video {video_path}: {e}")
        # 2. Delete GPS log (naming: same stem as video + _gps.json)
        gps_p = video_p.parent / f"{video_p.stem}_gps.json"
        if gps_p.is_file():
            try:
                gps_p.unlink()
                log.info(f"[TrailerVisionApp] Deleted GPS log: {gps_p.name}")
            except Exception as e:
                log.warning(f"[TrailerVisionApp] Failed to delete GPS log {gps_p}: {e}")
    
    def upload_to_server(self, video_path: str, crops_dir: str, data: Dict):
        """
        Optional hook for uploading data to AWS. Only video_processing type is sent here.
        Processed records (after data processor assigns spots) are uploaded by
        upload_processed_records_and_delete() in the periodic upload thread.
        """
        if data.get("type") == "ocr_complete":
            # Processed data is uploaded only after data processor runs; see upload_processed_records_and_delete
            return
        # Optional: handle type == 'video_processing' if needed
        if data.get("type") != "video_processing" or not data.get("results"):
            return
        log.debug("[TrailerVisionApp] Server upload hook: video_processing (optional)")
    
    def stop(self):
        """Stop the application and cleanup."""
        log.info("Stopping application...")
        self.running = False
        
        # Stop recording if active
        if self.video_recorder and self.video_recorder.is_recording():
            self.video_recorder.stop_recording()
        
        # Stop processing queue
        if self.processing_queue:
            self.processing_queue.stop()

        # Stop active vision pipeline
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception as e:
                log.warning("Error stopping vision pipeline: %s", e)
        
        # Stop recordings watcher
        self._recordings_watcher_stop.set()
        if self._recordings_watcher_thread and self._recordings_watcher_thread.is_alive():
            self._recordings_watcher_thread.join(timeout=5.0)
        
        # Stop GPS sensor
        if self.gps_sensor:
            self.gps_sensor.stop()
        
        # Stop publishers
        if self.publisher:
            self.publisher.stop()
        
        # Stop uploader
        if self.uploader:
            self.uploader.stop()
        
        # Close CSV logger
        if self.csv_logger:
            self.csv_logger.close()
        
        # Stop metrics server
        if self.metrics_server:
            self.metrics_server.stop()
        
        # Stop upload processed-records thread
        if getattr(self, "_upload_thread_stop", None) is not None:
            self._upload_thread_stop.set()
        if getattr(self, "_upload_thread", None) is not None and self._upload_thread.is_alive():
            self._upload_thread.join(timeout=5.0)
        
        log.info("Application stopped.")

    def get_default_gatevision_test_camera_id(self) -> str:
        """First configured gate_* camera id, or a stable synthetic id for file-based tests."""
        for c in self.config.get("cameras", []):
            role = str(c.get("role", "") or "").lower()
            if role.startswith("gate_"):
                cid = c.get("id")
                if cid:
                    return str(cid)
        return "gatevision_test"

    def get_gatevision_status(self) -> Dict:
        """Return GateVision runtime status and recent events for dashboard."""
        try:
            gate = self._get_gate_pipeline()
            if gate and hasattr(gate, "get_status"):
                status = gate.get_status()
            else:
                status = {"stats": {}, "recent_events": [], "config": {}}
        except Exception as e:
            status = {"stats": {}, "recent_events": [], "config": {}, "error": str(e)}
        cameras = self.config.get("cameras", [])
        gate_cameras = [
            {
                "id": c.get("id"),
                "role": c.get("role", "yard"),
                "gate_id": c.get("gate_id", "gate-1"),
            }
            for c in cameras
            if str(c.get("role", "")).startswith("gate_")
        ]
        status["gate_cameras"] = gate_cameras
        status["detection_mode"] = getattr(self, "detection_mode", "trailer")
        status["vision_pipeline"] = getattr(self.pipeline, "vision_mode", "hybrid")
        return status

    def update_gatevision_config(self, cfg: Dict) -> Dict:
        gate = self._get_gate_pipeline()
        if not gate or not hasattr(gate, "update_runtime_config"):
            return {"success": False, "message": "GateVision pipeline not available"}
        applied = gate.update_runtime_config(cfg or {})
        return {"success": True, "config": applied}

    def reload_gatevision_config_from_file(self) -> Dict:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            self.config = loaded
            g = self.config.get("globals", {}) or {}
            file_vm = str(g.get("vision_pipeline", "hybrid")).strip().lower()
            active_vm = getattr(self.pipeline, "vision_mode", "hybrid")
            if file_vm in ("yard", "gate", "hybrid") and file_vm != active_vm:
                log.info(
                    "[TrailerVisionApp] cameras.yaml has vision_pipeline=%s but active is %s — restart app to apply.",
                    file_vm,
                    active_vm,
                )
            gate_cfg = (self.config.get("gatevision", {}) or {})
            result = self.update_gatevision_config(gate_cfg)
            if result.get("success"):
                return {"success": True, "config": result.get("config", {}), "message": "GateVision config reloaded from file"}
            return result
        except Exception as e:
            return {"success": False, "message": str(e)}

    def reset_gatevision_runtime_state(self) -> Dict:
        gate = self._get_gate_pipeline()
        if not gate or not hasattr(gate, "reset_runtime_state"):
            return {"success": False, "message": "GateVision pipeline not available"}
        gate.reset_runtime_state()
        return {"success": True, "message": "GateVision runtime state reset"}

    def review_gatevision_event(self, gate_pass_id: str, decision: str, reason: str = "") -> Dict:
        gate = self._get_gate_pipeline()
        if not gate or not hasattr(gate, "review_event"):
            return {"success": False, "message": "GateVision pipeline not available"}
        return gate.review_event(gate_pass_id=gate_pass_id, decision=decision, reason=reason)

    def _get_gate_pipeline(self):
        if self.pipeline and hasattr(self.pipeline, "gate_pipeline"):
            return getattr(self.pipeline, "gate_pipeline")
        return None

    def store_gatevision_record_in_db(self, record: Dict) -> None:
        """
        Store GateVision detection/event evidence in local SQLite DB.
        Mirrors YardVision local persistence flow so GateVision detections
        are available for downstream processing/upload.
        """
        if not self.video_frame_db:
            return
        try:
            trailer_raw = (record.get("trailer_id") or "UNKNOWN").strip()
            from app.ocr.plate_validation import is_placeholder_value, sanitize_plate_for_storage
            
            if is_placeholder_value(trailer_raw):
                trailer_id = "UNKNOWN"
            else:
                try:
                    from app.container_utils import _extract_trailer_and_scac, _clean_trailer_number
                    trailer_part, scac_part = _extract_trailer_and_scac(trailer_raw)
                    cleaned = _clean_trailer_number(trailer_part)
                    if cleaned and cleaned != "UNKNOWN":
                        scac_clean = (scac_part or "").strip().upper()
                        trailer_id = f"{scac_clean}{cleaned}" if scac_clean else cleaned
                    else:
                        trailer_id = sanitize_plate_for_storage(trailer_raw) or "UNKNOWN"
                except Exception:
                    trailer_id = sanitize_plate_for_storage(trailer_raw) or "UNKNOWN"
            
            trailer_id = trailer_id[:50]
            timestamp_raw = record.get("timestamp")
            ts = None
            if timestamp_raw:
                try:
                    ts = datetime.fromisoformat(str(timestamp_raw).replace("Z", "+00:00"))
                except Exception:
                    ts = None
            self.video_frame_db.insert_gatevision_event(
                licence_plate_trailer=trailer_id,
                latitude=None,
                longitude=None,
                speed=None,
                confidence=float(record.get("confidence") or 0.0),
                image_path=record.get("image_path") or "",
                camera_id=record.get("camera_id") or "",
                frame_number=int(record.get("frame_number") or 0),
                track_id=record.get("track_id"),
                timestamp=ts,
                gate_id=record.get("gate_id") or "gate-1",
                event_type=record.get("stage") or "candidate",
                source="live",
                test_video_stem=None,
            )
        except Exception as e:
            log.info("[TrailerVisionApp] Failed to store GateVision record in DB: %s", e)


def main():
    """Main entry point."""
    setup_logging()
    # Initialize CUDA context in main thread before loading TensorRT engines
    # This ensures all worker threads can access the same context
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # Get the primary context created by autoinit
        primary_ctx = cuda.Context.get_current()
        if primary_ctx is not None:
            # Store in both module names for compatibility
            if '__main__' in sys.modules:
                sys.modules['__main__']._cuda_primary_context = primary_ctx
            # Also store in the actual module
            if 'app.main_trt_demo' in sys.modules:
                sys.modules['app.main_trt_demo']._cuda_primary_context = primary_ctx
            # Also store in this module's globals
            globals()['_cuda_primary_context'] = primary_ctx
            log.info("CUDA context initialized in main thread (via autoinit), context: %s", primary_ctx)
        else:
            log.warning("CUDA context initialization returned None")
    except Exception as e:
        log.warning("Failed to initialize CUDA context: %s", e)
        # Continue anyway - might work with autoinit in worker threads
    
    app = TrailerVisionApp()
    
    # Handle signals
    def signal_handler(sig, frame):
        app.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run application
    app.run()


if __name__ == '__main__':
    main()

