"""
Metrics Server with Web Dashboard

Flask server exposing JSON metrics, Prometheus metrics, health check,
events API, and static web dashboard files.
"""

import os
import re
import json
import cv2
import mimetypes
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from flask import Flask, jsonify, send_from_directory, request, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from werkzeug.utils import secure_filename
import threading
import time

from app.app_logger import get_logger, setup_logging

log = get_logger(__name__)


def _resolve_recordings_directory() -> Optional[Path]:
    for folder_name in ("out/recordings", "out/recording"):
        p = Path(folder_name)
        if p.exists():
            return p
    return None


def _list_recordings_video_files(recordings_dir: Path) -> List[Path]:
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    found: List[Path] = []
    for ext in video_extensions:
        found.extend(recordings_dir.glob(f"*{ext}"))
        found.extend(recordings_dir.rglob(f"*{ext}"))
    return sorted(set(found), key=lambda x: str(x).lower())


def _sanitize_gate_test_video_stem(stem: str) -> str:
    """Safe segment for SQLite video_path gatevision:test-<stem>:gate_pass."""
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", (stem or "clip").strip())
    s = s.strip("-")[:56] or "clip"
    return s


def _ensure_file_video_processing_queue(app, defer_ocr: bool = False, on_video_queue_drained=None):
    """
    Return ``app.processing_queue`` for file-based tests.

    Args:
        app: TrailerVisionApp instance (carries video_processor, ocr, etc.).
        defer_ocr: If True, the gate-test "YOLO first → unload YOLO → load OCR
            → run OCR" flow. The queue is created with ``defer_ocr=True`` and
            the caller MUST supply ``on_video_queue_drained`` (which loads OCR
            and queues OCR jobs) plus call ``pq.notify_recording_stopped()``
            after queueing all videos so the drain callback fires. OCR is NOT
            preloaded here in this mode — that's the whole point: we never
            want YOLO and Qwen-VL resident at the same time on Jetson.
        on_video_queue_drained: Callback invoked by the video worker after the
            video queue drains (only used when ``defer_ocr=True``).

    The legacy per-chunk-OCR path is preserved when ``defer_ocr=False``.
    """
    if not hasattr(app, "video_processor") or not app.video_processor:
        return None, (
            jsonify(
                {
                    "success": False,
                    "error": "Video processor not available",
                    "hint": "Start the application (Automatic tab) to initialize the video processor, or check logs.",
                }
            ),
            503,
        )
    # In legacy per-chunk-OCR mode the helper eagerly loads OCR. In defer_ocr
    # mode we deliberately skip that — OCR will be loaded by the drain
    # callback after the video phase has finished and YOLO is unloaded.
    if not defer_ocr and not getattr(app, "ocr", None):
        if hasattr(app, "_initialize_ocr"):
            app._initialize_ocr()
        if not app.ocr:
            return None, (
                jsonify(
                    {
                        "success": False,
                        "error": "OCR not available",
                        "hint": "Start the application to load OCR models.",
                    }
                ),
                503,
            )
    if getattr(app, "processing_queue", None):
        existing_defer = bool(getattr(app.processing_queue, "defer_ocr", False))
        if defer_ocr != existing_defer:
            log.info(
                "[MetricsServer] Stopping existing processing queue (defer_ocr=%s) to recreate with defer_ocr=%s",
                existing_defer,
                defer_ocr,
            )
            try:
                app.processing_queue.stop()
            except Exception as e:
                log.warning("[MetricsServer] Error stopping existing queue: %s", e)
            app.processing_queue = None

    if getattr(app, "processing_queue", None):
        # In defer_ocr mode the caller will (re)wire the drain callback per
        # request so a fresh batch's OCR step targets the latest jobs.
        if defer_ocr and on_video_queue_drained is not None:
            app.processing_queue.on_video_queue_drained = on_video_queue_drained
        if hasattr(app, "_sync_offline_gpu_lock_to_gate_pipeline"):
            app._sync_offline_gpu_lock_to_gate_pipeline()
        return app.processing_queue, None
    try:
        from app.processing_queue import ProcessingQueueManager

        app.processing_queue = ProcessingQueueManager(
            video_processor=app.video_processor,
            ocr=getattr(app, "ocr", None),  # may be None when defer_ocr=True
            preprocessor=getattr(app, "preprocessor", None),
            on_video_complete=None,
            on_ocr_complete=None,
            defer_ocr=defer_ocr,
            on_video_queue_drained=on_video_queue_drained,
        )
        log.info(
            "[MetricsServer] Created processing queue for GateVision file tests (defer_ocr=%s)",
            defer_ocr,
        )
        if hasattr(app, "_sync_offline_gpu_lock_to_gate_pipeline"):
            app._sync_offline_gpu_lock_to_gate_pipeline()
        return app.processing_queue, None
    except Exception as e:
        log.exception("[MetricsServer] Failed to create processing queue: %s", e)
        return None, (jsonify({"success": False, "error": str(e)}), 500)


# Prometheus metrics
frames_processed = Counter('trailer_vision_frames_processed_total', 'Total frames processed', ['camera_id'])
detections_total = Counter('trailer_vision_detections_total', 'Total detections', ['camera_id'])
events_logged = Counter('trailer_vision_events_logged_total', 'Total events logged')
fps_gauge = Gauge('trailer_vision_fps', 'Frames per second (EMA)', ['camera_id'])
queue_depth_gauge = Gauge('trailer_vision_queue_depth', 'Event queue depth')
last_publish_time = Gauge('trailer_vision_last_publish_timestamp', 'Last publish timestamp', ['camera_id'])

# Application metrics registry
metrics_registry = {
    'cameras': {}  # camera_id -> {fps_ema, frames_processed, last_publish, queue_depth}
}


class MetricsServer:
    """
    Metrics server with web dashboard.
    """
    
    def __init__(self, port: int = 8080, csv_logger=None, frame_storage=None, video_processor=None):
        """
        Initialize metrics server.
        
        Args:
            port: HTTP server port
            csv_logger: Optional CSV logger for events API
            frame_storage: Optional frame storage object (TrailerVisionApp instance)
            video_processor: Optional video processor instance
        """
        setup_logging()
        self.port = port
        self.csv_logger = csv_logger
        self.frame_storage = frame_storage
        self.video_processor = video_processor
        self.app = Flask(__name__, static_folder='../web', static_url_path='')
        self.upload_dir = Path(__file__).resolve().parent.parent / "out" / "recordings"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self._setup_error_handlers()
        self._setup_routes()
        self.thread = None
        self.running = False
        
        # Video upload directory
        self.upload_dir = Path(tempfile.gettempdir()) / 'trailer_vision_uploads'
        self.upload_dir.mkdir(exist_ok=True)
        
        # Processing status tracking
        self.processing_status = {
            'status': 'idle',  # 'idle', 'processing_video', 'processing_ocr', 'completed', 'error'
            'message': '',
            'video_processing_complete': False,
            'ocr_processing_complete': False
        }
        self.status_lock = threading.Lock()
        

    
    def _setup_error_handlers(self):
        """Setup Flask error handlers."""
        from werkzeug.exceptions import NotFound
        
        @self.app.errorhandler(NotFound)
        def handle_not_found(e):
            """Handle 404 Not Found errors gracefully."""
            # Don't log 404s for static files as errors - they're expected
            return jsonify({'error': 'Not found'}), 404
        
        @self.app.errorhandler(Exception)
        def handle_exception(e):
            """Handle all other exceptions."""
            # Skip logging for NotFound exceptions (already handled above)
            from werkzeug.exceptions import NotFound
            if isinstance(e, NotFound):
                return jsonify({'error': 'Not found'}), 404
            
            import traceback
            error_msg = str(e)
            traceback_str = traceback.format_exc()
            log.exception("Unhandled Exception: %s", error_msg)
            return jsonify({'error': 'Internal server error', 'message': error_msg}), 500
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        # API routes must be defined BEFORE catch-all routes
        # to ensure they are matched first
        
        @self.app.route('/metrics.json')
        def metrics_json():
            """Get metrics as JSON."""
            return jsonify(self._get_metrics())
        
        @self.app.route('/metrics')
        def metrics_prometheus():
            """Get metrics in Prometheus format."""
            return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
        
        @self.app.route('/healthz')
        def healthz():
            """Health check endpoint."""
            return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})
        
        @self.app.route('/events')
        def events():
            """Get recent events from CSV."""
            camera_id = request.args.get('camera_id')
            limit = int(request.args.get('limit', 100))
            
            events = self._get_recent_events(camera_id, limit)
            return jsonify({'events': events, 'count': len(events)})
        
        @self.app.route('/stream/<camera_id>')
        def stream_camera(camera_id):
            """MJPEG stream for a camera."""
            return Response(
                self._generate_mjpeg(camera_id),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @self.app.route('/api/start-recording', methods=['POST'])
        def start_recording():
            """Start video recording with GPS logging."""
            try:
                data = request.get_json() or {}
                camera_id = data.get('camera_id')
                
                # Get app instance from frame_storage
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'start_recording'):
                    return jsonify({'error': 'Recording not available'}), 500
                
                result = app.start_recording(camera_id=camera_id)
                
                if result.get('success'):
                    return jsonify(result), 200
                else:
                    return jsonify(result), 400
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/stop-recording', methods=['POST'])
        def stop_recording():
            """Stop video recording and return paths."""
            try:
                # Get app instance from frame_storage
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'stop_recording'):
                    return jsonify({'error': 'Recording not available'}), 500
                
                result = app.stop_recording()
                
                if result.get('success'):
                    return jsonify(result), 200
                else:
                    return jsonify(result), 400
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/recording-status', methods=['GET'])
        def recording_status():
            """Get current recording status."""
            try:
                # Get app instance from frame_storage
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'is_recording'):
                    return jsonify({'recording': False, 'gps_available': False, 'error': 'Recording not available'}), 200
                
                is_recording = app.is_recording()
                gps_available = getattr(app, 'gps_sensor', None) is not None
                payload = {'recording': is_recording, 'gps_available': gps_available}
                if is_recording and getattr(app, 'video_recorder', None):
                    camera_id = getattr(app.video_recorder, 'camera_id', None)
                    if camera_id:
                        payload['camera_id'] = camera_id
                return jsonify(payload), 200
            except Exception as e:
                return jsonify({'recording': False, 'gps_available': False, 'error': str(e)}), 200
        
        @self.app.route('/api/upload-video', methods=['POST'])
        def upload_video():
            """Upload video file for processing & perform instant in-memory mid-frame OCR."""
            if 'video' not in request.files:
                return jsonify({'error': 'No video file provided'}), 400
            
            file = request.files['video']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Clean up older uploaded video files from out/recordings
            filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())
            upload_dir = Path(__file__).resolve().parent.parent / "out" / "recordings"
            upload_dir.mkdir(parents=True, exist_ok=True)
            try:
                for old_f in upload_dir.glob("*.*"):
                    if old_f.is_file():
                        old_f.unlink(missing_ok=True)
                log.info("[MetricsServer] Cleared older uploaded video files from out/recordings")
            except Exception as clean_err:
                log.warning(f"[MetricsServer] Pre-upload cleanup notice: {clean_err}")

            file_path = upload_dir / f"{unique_id}_{filename}"
            file.save(str(file_path))
            log.info("[MetricsServer] Video file uploaded and saved: %s", file_path.name)

            return jsonify({
                'success': True,
                'video_id': file_path.name,
                'filename': filename,
                'video_path': str(file_path),
                'status': 'uploaded',
                'ocr_result': {"container_number": "", "confidence": 0.0, "image_path": ""},
                'message': f'Video uploaded successfully ({filename}). Click Process Video to start.'
            }), 200
        
        @self.app.route('/api/process-video', methods=['POST'])
        @self.app.route('/api/process-video/<path:video_id>', methods=['POST'])
        def process_video(video_id=None):
            """Start processing uploaded video."""
            if not video_id and request.is_json:
                req_data = request.get_json(silent=True) or {}
                video_id = req_data.get('video_id') or req_data.get('video_path')
            if not video_id:
                return jsonify({'error': 'No video selected or uploaded yet.'}), 400

            log.info(f"[MetricsServer] process_video called with video_id: {video_id}")
            try:
                log.info(f"[MetricsServer] Checking video_processor: {self.video_processor is not None}")
                if self.video_processor is None:
                    log.error("Video processor not available")
                    return jsonify({'error': 'Video processor not available'}), 500
                
                # Reset video processor state for new processing request
                try:
                    if self.video_processor.is_processing():
                        log.info("[MetricsServer] Resetting video processor state for new processing request...")
                        self.video_processor.stop_processing()
                        with self.video_processor.lock:
                            self.video_processor.processing = False
                except Exception as e:
                    log.warning("Notice resetting video processor state: %s", e)
                
                # Find video file across all possible directory layouts
                try:
                    log.info(f"[MetricsServer] Looking for video file with ID/path: {video_id}")
                    video_files = []
                    
                    # 1. Direct path check
                    if video_id and Path(video_id).exists() and Path(video_id).is_file():
                        video_files = [Path(video_id)]
                    
                    # 2. Search in recordings directories
                    if not video_files:
                        dirs_to_search = [
                            getattr(self, 'upload_dir', None),
                            Path("out/recordings").resolve(),
                            Path(__file__).resolve().parent.parent / "out" / "recordings"
                        ]
                        for d in dirs_to_search:
                            if d and d.exists():
                                matches = list(d.glob(f"{video_id}_*")) or list(d.glob(f"{video_id}*"))
                                if not matches and (d / video_id).exists():
                                    matches = [d / video_id]
                                if matches:
                                    video_files = matches
                                    break

                    log.info(f"[MetricsServer] Found {len(video_files)} matching files for ID/path {video_id}")
                    if not video_files:
                        log.error("Video file not found for ID: %s", video_id)
                        return jsonify({'error': 'Video file not found. Please re-select and upload your video file.'}), 404
                    
                    video_path = str(video_files[0])
                    log.info(f"[MetricsServer] Resolved video file path: {video_path}")
                    
                    # Look for corresponding GPS log file
                    video_path_obj = Path(video_path)
                    gps_log_path = None
                    
                    # Extract camera ID, date, and time from video filename
                    # Format: {video_id}_{camera_id}_{date}_{time}.mp4
                    # Example: 5b8e7e9a-b5f8-4ff2-ad12-b7f9b837b223_lifecam-hd6000-01_20260116_174341.mp4
                    video_stem = video_path_obj.stem
                    parts = video_stem.split('_')
                    camera_id = None
                    date_str = None
                    time_str = None
                    
                    if len(parts) >= 3:
                        # Try to find camera_id, date, and time in filename
                        # Look for pattern: {video_id}_{camera_id}_{date}_{time}
                        # Find the date part (8 digits: YYYYMMDD) and time part (6 digits: HHMMSS)
                        for i in range(1, len(parts) - 1):
                            # Check if this part looks like a date (YYYYMMDD)
                            if len(parts[i]) == 8 and parts[i].isdigit():
                                # Camera ID is everything before the date
                                camera_id = '_'.join(parts[1:i])  # Skip video_id (parts[0])
                                date_str = parts[i]
                                # Check if next part is time (6 digits: HHMMSS)
                                if i + 1 < len(parts) and len(parts[i+1]) == 6 and parts[i+1].isdigit():
                                    time_str = parts[i+1]
                                break
                    
                    # Try to find GPS log with same base name (in same directory)
                    possible_gps_paths = [
                        video_path_obj.parent / f"{video_path_obj.stem}_gps.json",
                        video_path_obj.parent / f"{video_path_obj.stem.replace('_', '_')}_gps.json"
                    ]
                    
                    # Also check out/recordings directory for GPS log files
                    recordings_dir = Path("out/recordings")
                    if recordings_dir.exists() and camera_id and date_str:
                        # Priority 1: Try exact match with time component
                        if time_str:
                            exact_match = recordings_dir / f"{camera_id}_{date_str}_{time_str}_gps.json"
                            if exact_match.exists():
                                possible_gps_paths.insert(0, exact_match)  # Highest priority
                                log.info(f"[MetricsServer] Found exact GPS log match: {exact_match}")
                            else:
                                # Priority 2: Find closest timestamp match
                                # Look for all GPS log files matching camera_id and date
                                gps_pattern = f"{camera_id}_{date_str}_*_gps.json"
                                matching_gps_files = list(recordings_dir.glob(gps_pattern))
                                if matching_gps_files:
                                    # Extract timestamps and find the closest match
                                    def extract_time_from_filename(filename):
                                        """Extract time string (HHMMSS) from GPS log filename."""
                                        name = filename.stem  # Remove .json extension
                                        parts = name.split('_')
                                        # Find the part that looks like time (6 digits)
                                        for part in parts:
                                            if len(part) == 6 and part.isdigit():
                                                return part
                                        return None
                                    
                                    # Convert video time to seconds for comparison
                                    video_time_sec = int(time_str[:2]) * 3600 + int(time_str[2:4]) * 60 + int(time_str[4:6])
                                    
                                    best_match = None
                                    min_time_diff = float('inf')
                                    
                                    for gps_file in matching_gps_files:
                                        gps_time_str = extract_time_from_filename(gps_file)
                                        if gps_time_str:
                                            gps_time_sec = int(gps_time_str[:2]) * 3600 + int(gps_time_str[2:4]) * 60 + int(gps_time_str[4:6])
                                            time_diff = abs(gps_time_sec - video_time_sec)
                                            if time_diff < min_time_diff:
                                                min_time_diff = time_diff
                                                best_match = gps_file
                                    
                                    if best_match:
                                        # Only use if within 5 minutes (300 seconds)
                                        if min_time_diff <= 300:
                                            possible_gps_paths.insert(0, best_match)
                                            log.info(f"[MetricsServer] Found closest GPS log match (time diff: {min_time_diff}s): {best_match}")
                                        else:
                                            log.info(f"[MetricsServer] GPS log files found but time difference too large ({min_time_diff}s > 300s), using closest anyway")
                                            possible_gps_paths.append(best_match)
                                    else:
                                        # Fallback: use first matching file
                                        possible_gps_paths.append(matching_gps_files[0])
                                        log.info(f"[MetricsServer] Found GPS log in recordings directory (no time match): {matching_gps_files[0]}")
                                else:
                                    # Try without time component (just camera_id and date)
                                    gps_pattern_alt = f"{camera_id}_{date_str}_gps.json"
                                    matching_gps_files_alt = list(recordings_dir.glob(gps_pattern_alt))
                                    if matching_gps_files_alt:
                                        possible_gps_paths.append(matching_gps_files_alt[0])
                                        log.info(f"[MetricsServer] Found GPS log in recordings directory (no time): {matching_gps_files_alt[0]}")
                        else:
                            # No time component in video filename, just match by camera_id and date
                            gps_pattern = f"{camera_id}_{date_str}_*_gps.json"
                            matching_gps_files = list(recordings_dir.glob(gps_pattern))
                            if matching_gps_files:
                                possible_gps_paths.append(matching_gps_files[0])
                                log.info(f"[MetricsServer] Found GPS log in recordings directory: {matching_gps_files[0]}")
                            else:
                                # Try without time component
                                gps_pattern_alt = f"{camera_id}_{date_str}_gps.json"
                                matching_gps_files_alt = list(recordings_dir.glob(gps_pattern_alt))
                                if matching_gps_files_alt:
                                    possible_gps_paths.append(matching_gps_files_alt[0])
                                    log.info(f"[MetricsServer] Found GPS log in recordings directory (no time): {matching_gps_files_alt[0]}")
                    
                    # Try all possible paths in priority order
                    for gps_path in possible_gps_paths:
                        if gps_path.exists():
                            gps_log_path = str(gps_path)
                            log.info(f"[MetricsServer] Found GPS log file: {gps_log_path}")
                            break
                    
                    if not gps_log_path:
                        log.info(f"[MetricsServer] No GPS log file found for video (this is OK if video was not recorded with GPS)")
                        log.info(f"[MetricsServer] Searched in: {video_path_obj.parent} and {recordings_dir if recordings_dir.exists() else 'N/A'}")
                        if camera_id and date_str:
                            log.info(f"[MetricsServer] Looking for GPS log with camera_id={camera_id}, date={date_str}, time={time_str if time_str else 'N/A'}")
                except Exception as e:
                    log.error("Failed to find video file: %s", e)
                    import traceback
                    traceback.print_exc()
                    return jsonify({'error': f'Failed to find video file: {str(e)}'}), 500
                
                # Get detect_every_n and detection_mode parameters safely from JSON, args, or form
                detect_every_n = 5
                detection_mode = 'gate'  # default fallback
                try:
                    # 1. Try Query string parameter
                    if request.args.get('detection_mode'):
                        detection_mode = str(request.args.get('detection_mode')).lower()
                    # 2. Try JSON payload
                    elif request.is_json:
                        json_data = request.get_json(silent=True) or {}
                        if json_data.get('detection_mode'):
                            detection_mode = str(json_data['detection_mode']).lower()
                        if json_data.get('detect_every_n'):
                            detect_every_n = int(json_data['detect_every_n'])
                    # 3. Try Form parameter
                    elif request.form.get('detection_mode'):
                        detection_mode = str(request.form.get('detection_mode')).lower()

                    if detection_mode not in ['gate', 'stacker']:
                        detection_mode = 'gate'
                    log.info(f"[MetricsServer] Selected detection_mode for video run: {detection_mode}")
                except Exception as e:
                    log.warning(f"[MetricsServer] Parameter parse notice: {e}")
                
                # Capture video_path, GPS log path, detect_every_n, and detection_mode for background thread
                # (Flask request context is not available in background threads)
                captured_video_path = video_path
                captured_gps_log_path = gps_log_path if 'gps_log_path' in locals() else None
                captured_detect_every_n = detect_every_n
                captured_detection_mode = detection_mode
                
                # Clear stale records from database before processing new video
                try:
                    import sqlite3
                    conn = sqlite3.connect("data/inventory.db")
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM yardvision_records")
                    cursor.execute("DELETE FROM video_frame_records")
                    conn.commit()
                    conn.close()
                    log.info("[MetricsServer] Cleared stale local records for clean video run.")
                except Exception as db_err:
                    log.warning("[MetricsServer] DB cleanup notice: %s", db_err)

                # Reset status before starting new processing
                with self.status_lock:
                    self.processing_status['status'] = 'idle'
                    self.processing_status['message'] = ''
                    self.processing_status['video_processing_complete'] = False
                    self.processing_status['ocr_processing_complete'] = False
                
                # Start processing in background thread
                def process_in_background():
                    try:
                        import traceback
                        log.info(f"[MetricsServer] Starting video processing: {captured_video_path}, detect_every_n={captured_detect_every_n}, detection_mode={captured_detection_mode}")
                        
                        # Update status: video processing started
                        with self.status_lock:
                            self.processing_status['status'] = 'processing_video'
                            self.processing_status['message'] = f'Processing video ({captured_detection_mode} mode)...'
                            self.processing_status['video_processing_complete'] = False
                            self.processing_status['ocr_processing_complete'] = False
                        
                        log.info(f"[MetricsServer] Setting detection_mode={captured_detection_mode} on video_processor")
                        if hasattr(self.video_processor, 'set_detection_mode'):
                            self.video_processor.set_detection_mode(captured_detection_mode)

                        original_detector = self.video_processor.detector
                        try:
                            from app.ai.factory import get_detector
                            log.info(f"[MetricsServer] Loading detector via AI Factory for mode: '{captured_detection_mode}'")
                            self.video_processor.detector = get_detector(
                                mode=captured_detection_mode,
                                conf_threshold=0.25,
                                fallback_detector=original_detector
                            )
                        except Exception as e:
                            log.warning(f"[MetricsServer] Detector setup notice for {captured_detection_mode} mode: {e}")
                        except Exception as e:
                            log.info(f"[MetricsServer] Warning: Detector setup for {captured_detection_mode} mode: {e}")

                        frame_count = 0
                        last_results_check = 0

                        # Update video processor with GPS log path if available
                        if captured_gps_log_path and hasattr(self.video_processor, 'gps_log_path'):
                            self.video_processor.gps_log_path = captured_gps_log_path
                            if hasattr(self.video_processor, 'gps_log'):
                                from app.container_utils import load_gps_log
                                self.video_processor.gps_log = load_gps_log(captured_gps_log_path)
                                log.info(f"[MetricsServer] Loaded GPS log for video processing: {captured_gps_log_path}")

                        for frame_num, processed_frame, events in self.video_processor.process_video(
                            captured_video_path, camera_id="gate-video", detect_every_n=captured_detect_every_n
                        ):
                            frame_count += 1

                            # Check if we should stop
                            if not self.video_processor.is_processing():
                                log.info(f"[MetricsServer] Processing stopped at frame {frame_num}")
                                break

                            # Log progress every 30 frames
                            if frame_count - last_results_check >= 30:
                                results = self.video_processor.get_results()
                                log.info(f"[MetricsServer] Progress: {results['frames_processed']} frames, {results['detections']} detections, {results['tracks']} tracks")
                                last_results_check = frame_count

                        # Video frame loop completed
                        final_results = self.video_processor.get_results()
                        log.info(f"[MetricsServer] Video processing completed: {frame_count} frames processed. Final results: {final_results}")

                        # Restore original detector
                        if 'original_detector' in locals() and original_detector is not None:
                            self.video_processor.detector = original_detector
                            log.info(f"[MetricsServer] Restored original detector")

                        # Get crops directory for OCR
                        crops_dir = None
                        if hasattr(self.video_processor, 'crops_dir') and self.video_processor.crops_dir:
                            crops_dir = str(self.video_processor.crops_dir)

                        import os
                        if crops_dir and os.path.exists(crops_dir):
                            log.info(f"[MetricsServer] Video processing complete. Found crops_dir: {crops_dir}. Automating Gate OCR...")

                            app_ref = self.frame_storage if hasattr(self, 'frame_storage') else None
                            if app_ref:
                                with self.status_lock:
                                    self.processing_status['status'] = 'processing_ocr'
                                    self.processing_status['message'] = 'Running OCR on container crops...'

                                if hasattr(app_ref, "_initialize_ocr") and not getattr(app_ref, "ocr", None):
                                    app_ref._initialize_ocr()

                                if getattr(app_ref, "ocr", None):
                                    try:
                                        from app.batch_ocr_processor import BatchOCRProcessor
                                        batch_processor = BatchOCRProcessor(app_ref.ocr, getattr(app_ref, 'preprocessor', None))
                                        ocr_results = batch_processor.process_crops_directory(
                                            crops_dir, should_stop=lambda: getattr(self.video_processor, 'stop_flag', False)
                                        )
                                        combined_results = batch_processor.match_ocr_to_detections(crops_dir, ocr_results)
                                        if combined_results and hasattr(app_ref, '_store_ocr_results_in_db'):
                                            stem = Path(captured_video_path).stem
                                            db_vp = f"gate:video-{stem}:gate_pass"
                                            app_ref._store_ocr_results_in_db(
                                                captured_video_path,
                                                crops_dir,
                                                combined_results,
                                                video_path_db=db_vp,
                                                allow_missing_gps=True,
                                            )
                                            log.info("[MetricsServer] Stored Gate OCR results in database")
                                            if hasattr(app_ref, "upload_processed_records_and_delete"):
                                                app_ref.upload_processed_records_and_delete()
                                    except Exception as ocr_err:
                                        log.exception("[MetricsServer] Gate OCR failed: %s", ocr_err)

                        # Update status: completed
                        with self.status_lock:
                            self.processing_status['status'] = 'completed'
                            self.processing_status['message'] = 'Processing completed successfully (Video + OCR done).'
                            self.processing_status['video_processing_complete'] = True
                            self.processing_status['ocr_processing_complete'] = True

                        log.info(f"[MetricsServer] Gate Video process complete. Video + OCR finished successfully.")
                        
                    except Exception as e:
                        log.error("Error processing video: %s", e)
                        import traceback
                        traceback.print_exc()
                        # Restore original detector on error
                        if 'original_detector' in locals() and original_detector is not None:
                            try:
                                self.video_processor.detector = original_detector
                                log.info(f"[MetricsServer] Restored original detector after error")
                            except:
                                pass
                        # Make sure processing flag is cleared on error
                        try:
                            self.video_processor.stop_processing()
                        except:
                            pass
                        # Update status: error
                        with self.status_lock:
                            self.processing_status['status'] = 'error'
                            self.processing_status['message'] = f'Processing failed: {str(e)}'
                            self.processing_status['video_processing_complete'] = False
                            self.processing_status['ocr_processing_complete'] = False
                    finally:
                        pass
                
                thread = threading.Thread(target=process_in_background, daemon=True)
                thread.start()
                
                log.info(f"[MetricsServer] Video processing thread started for video_id: {video_id}")
                return jsonify({'status': 'processing_started', 'video_id': video_id})
                
            except Exception as e:
                log.exception("FATAL ERROR in process_video endpoint: %s", e)
                import traceback
                traceback.print_exc()
                return jsonify({'error': f'Internal server error: {str(e)}'}), 500
        
        @self.app.route('/api/stop-processing', methods=['POST'])
        def stop_processing():
            """Stop current video processing."""
            if self.video_processor is None:
                return jsonify({'error': 'Video processor not available'}), 500
            
            self.video_processor.stop_processing()
            return jsonify({'status': 'stopped'})
        
        @self.app.route('/api/processing-results', methods=['GET'])
        def get_processing_results():
            """Get current processing results."""
            if self.video_processor is None:
                return jsonify({'error': 'Video processor not available'}), 500
            
            results = self.video_processor.get_results()
            results['processing'] = self.video_processor.is_processing()
            
            # Add processing status
            with self.status_lock:
                results['processing_status'] = self.processing_status.copy()
            
            return jsonify(results)
        
        @self.app.route('/api/processing-status', methods=['GET'])
        def get_processing_status():
            """Get current processing status for dashboard."""
            with self.status_lock:
                return jsonify(self.processing_status.copy())
        
        # ========== DEBUG ENDPOINTS ==========
        
        @self.app.route('/api/debug/start-auto-recording', methods=['POST'])
        def debug_start_auto_recording():
            """Debug: Start auto recording with 45-second chunking."""
            try:
                data = request.get_json() or {}
                camera_id = data.get('camera_id')
                
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'start_recording'):
                    return jsonify({'error': 'Recording not available'}), 500
                
                result = app.start_recording(camera_id=camera_id)
                
                if result.get('success'):
                    return jsonify({
                        'success': True,
                        'message': 'Auto recording started with 45-second chunking',
                        'camera_id': result.get('camera_id')
                    }), 200
                else:
                    return jsonify(result), 400
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/debug/stop-auto-recording', methods=['POST'])
        def debug_stop_auto_recording():
            """Debug: Stop auto recording."""
            try:
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'stop_recording'):
                    return jsonify({'error': 'Recording not available'}), 500
                
                result = app.stop_recording()
                
                if result.get('success'):
                    return jsonify({
                        'success': True,
                        'message': 'Auto recording stopped',
                        'result': result
                    }), 200
                else:
                    return jsonify(result), 400
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/debug/stop-video-processing', methods=['POST'])
        def debug_stop_video_processing():
            """Debug: Stop current video processing and clear pending video jobs."""
            try:
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'video_processor'):
                    return jsonify({'error': 'Video processor not available'}), 500
                
                if not app.video_processor:
                    return jsonify({'error': 'Video processor not initialized'}), 500
                
                # Stop current video job (sets stop flag so loop exits)
                if hasattr(app.video_processor, 'stop_processing'):
                    app.video_processor.stop_processing()
                
                # Clear pending video jobs so they are not processed
                if hasattr(app, 'processing_queue') and app.processing_queue:
                    app.processing_queue.clear_video_queue()
                
                return jsonify({
                    'success': True,
                    'message': 'Video processing stopped'
                }), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/debug/stop-ocr-processing', methods=['POST'])
        def debug_stop_ocr_processing():
            """Debug: Stop current OCR processing and clear OCR queue."""
            try:
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'processing_queue'):
                    return jsonify({'error': 'Processing queue not available'}), 500
                
                if not app.processing_queue:
                    return jsonify({'error': 'Processing queue not initialized'}), 500
                
                # Request current OCR job to stop (worker checks this between crops)
                app.processing_queue.request_ocr_stop()
                # Clear pending OCR jobs
                cleared_jobs = app.processing_queue.clear_ocr_queue()
                
                return jsonify({
                    'success': True,
                    'message': f'OCR processing stopped. Queue cleared ({cleared_jobs} job(s) removed).',
                    'cleared_jobs': cleared_jobs
                }), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/debug/start-video-processing', methods=['POST'])
        def debug_start_video_processing():
            """Debug: Manually trigger video processing on a video file or all videos in out/recordings."""
            try:
                data = request.get_json() or {}
                video_path = data.get('video_path')
                process_all = data.get('process_all', False)
                camera_id = data.get('camera_id', 'test-video')
                gps_log_path = data.get('gps_log_path')
                detection_mode = data.get('detection_mode', 'trailer')

                app = self.frame_storage if hasattr(self, 'frame_storage') else None

                # Read default ``detect_every_n`` from cameras.yaml so this debug
                # endpoint matches the auto-watcher cadence; explicit body wins.
                _gcfg = (getattr(app, 'config', {}) or {}).get('globals', {}) or {} if app else {}
                _default_den = int(_gcfg.get('detect_every_n', 20))
                detect_every_n = int(data.get('detect_every_n', _default_den))
                       # For debug mode, we need to ensure we have a processing queue
                # If app exists but processing queue doesn't, create one on the fly
                processing_queue = None
                
                if app:
                    if not hasattr(app, 'video_processor') or not app.video_processor:
                        return jsonify({
                            'error': 'Video processor not available',
                            'hint': 'Please start the application first to initialize the video processor.'
                        }), 500

                    # ─── Automated Debug Flow (Deferred OCR) ─────────────────────
                    # Unload YOLO -> load OCR -> run OCR -> restore YOLO automatically
                    # to prevent GPU memory OOM on Jetson.
                    _debug_batch_state = {
                        'remaining': 0,
                        'lock': threading.Lock(),
                    }

                    def _debug_post_cleanup():
                        app_ref = self.frame_storage if hasattr(self, 'frame_storage') else None
                        if app_ref is None:
                            return
                        try:
                            log.info(
                                "[MetricsServer] Debug post-cleanup: unloading OCR and restoring YOLO for live mode"
                            )
                            if hasattr(app_ref, "_unload_ocr"):
                                app_ref._unload_ocr()
                            if hasattr(app_ref, "_ensure_detector_loaded"):
                                app_ref._ensure_detector_loaded()
                        except Exception as ex:
                            log.exception("[MetricsServer] Debug post-cleanup failed: %s", ex)

                    def _on_debug_ocr_complete(video_path, crops_dir, ocr_results):
                        """Store OCR results in database when debug processing completes OCR for a job."""
                        app_ref = self.frame_storage if hasattr(self, 'frame_storage') else None
                        if not app_ref or not getattr(app_ref, 'video_frame_db', None) or not ocr_results:
                            pass
                        else:
                            try:
                                if hasattr(app_ref, '_store_ocr_results_in_db'):
                                    app_ref._store_ocr_results_in_db(video_path, crops_dir, ocr_results, allow_missing_gps=True)
                                    log.info("[MetricsServer] Debug OCR: stored results in database for %s", Path(video_path).name)
                            except Exception as e:
                                log.exception("[MetricsServer] Debug OCR: failed to store results in database: %s", e)
                        
                        is_last = False
                        with _debug_batch_state['lock']:
                            _debug_batch_state['remaining'] -= 1
                            is_last = _debug_batch_state['remaining'] <= 0
                        if is_last:
                            _debug_post_cleanup()

                    def _debug_drain_to_ocr_phase(pending_jobs):
                        """Swap GPU residency from YOLO to OCR, then queue and start the OCR jobs."""
                        app_ref = self.frame_storage if hasattr(self, 'frame_storage') else None
                        if app_ref is None:
                            return
                        try:
                            log.info(
                                "[MetricsServer] Debug transition: video phase done, swapping YOLO→OCR for %s job(s)",
                                len(pending_jobs),
                            )
                            if hasattr(app_ref, "_unload_detector"):
                                app_ref._unload_detector()
                            if not getattr(app_ref, "ocr", None) and hasattr(app_ref, "_initialize_ocr"):
                                app_ref._initialize_ocr()
                            if not getattr(app_ref, "ocr", None):
                                log.error("[MetricsServer] Debug transition: OCR failed to load; restoring YOLO")
                                _debug_post_cleanup()
                                return
                            pq_ref = getattr(app_ref, "processing_queue", None)
                            if pq_ref is None:
                                _debug_post_cleanup()
                                return
                            pq_ref.set_ocr(app_ref.ocr)
                            
                            annotated_jobs = []
                            for j in pending_jobs:
                                j2 = dict(j)
                                j2['on_ocr_complete'] = _on_debug_ocr_complete
                                annotated_jobs.append(j2)
                                
                            with _debug_batch_state['lock']:
                                _debug_batch_state['remaining'] = len(annotated_jobs)
                            pq_ref.queue_ocr_jobs(annotated_jobs)
                            pq_ref.start_ocr_worker_if_deferred()
                        except Exception as ex:
                            log.exception("[MetricsServer] Debug transition failed: %s", ex)
                            _debug_post_cleanup()

                    # Always use or create a processing queue with defer_ocr=True for Jetson memory safety & automation
                    if hasattr(app, 'processing_queue') and app.processing_queue and getattr(app.processing_queue, 'defer_ocr', False):
                        processing_queue = app.processing_queue
                        processing_queue.on_ocr_complete = _on_debug_ocr_complete
                        processing_queue.on_video_queue_drained = _debug_drain_to_ocr_phase
                    else:
                        log.info("[MetricsServer] Creating debug processing queue (deferred OCR mode for automation)")
                        try:
                            if getattr(app, 'processing_queue', None):
                                try:
                                    app.processing_queue.stop()
                                except:
                                    pass
                                app.processing_queue = None
                                
                            from app.processing_queue import ProcessingQueueManager
                            processing_queue = ProcessingQueueManager(
                                video_processor=app.video_processor,
                                ocr=None,  # Will load deferred
                                preprocessor=getattr(app, 'preprocessor', None),
                                on_video_complete=None,
                                on_ocr_complete=_on_debug_ocr_complete,
                                defer_ocr=True,
                                on_video_queue_drained=_debug_drain_to_ocr_phase
                            )
                            app.processing_queue = processing_queue
                            log.info("[MetricsServer] Debug processing queue created (deferred OCR)")
                            if hasattr(app, "_sync_offline_gpu_lock_to_gate_pipeline"):
                                app._sync_offline_gpu_lock_to_gate_pipeline()
                        except Exception as e:
                            log.info(f"[MetricsServer] Failed to create debug processing queue: {e}")
                            import traceback
                            traceback.print_exc()
                            return jsonify({
                                'error': f'Failed to initialize processing queue: {str(e)}',
                                'hint': 'Make sure the video processor and OCR are initialized (start the application).'
                            }), 500
                else:
                    # No app available - check if we can use video_processor from metrics_server
                    if self.video_processor:
                        return jsonify({
                            'error': 'Application not available. Cannot create processing queue without full application context.',
                            'hint': 'Please start the application first to initialize all components.'
                        }), 500
                    else:
                        return jsonify({
                            'error': 'Application not available. Please start the application first.',
                            'hint': 'Click "Start Application" button to initialize the processing queue.'
                        }), 500
                
                if not processing_queue:
                    return jsonify({
                        'error': 'Processing queue not available and could not be created.',
                        'hint': 'Please start the application first to initialize all components.'
                    }), 500

                # Reset deferred-OCR state for this batch
                if hasattr(processing_queue, 'notify_recording_started'):
                    processing_queue.notify_recording_started()
                
                # Process all videos in out/recordings folder (or out/recording)
                if process_all or not video_path:
                    # Try both singular and plural folder names
                    recordings_dir = None
                    for folder_name in ["out/recordings", "out/recording"]:
                        test_dir = Path(folder_name)
                        if test_dir.exists():
                            recordings_dir = test_dir
                            break
                    
                    if not recordings_dir or not recordings_dir.exists():
                        return jsonify({
                            'error': f'Recordings directory not found. Checked: out/recordings and out/recording',
                            'checked_paths': ['out/recordings', 'out/recording']
                        }), 404
                    
                    # Find all video files (search recursively in subdirectories too)
                    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
                    video_files = []
                    for ext in video_extensions:
                        # Search in root directory
                        video_files.extend(recordings_dir.glob(f'*{ext}'))
                        # Search recursively in subdirectories
                        video_files.extend(recordings_dir.rglob(f'*{ext}'))
                    
                    # Remove duplicates (in case same file matches both patterns)
                    video_files = list(set(video_files))
                    
                    log.info(f"[MetricsServer] Found {len(video_files)} video file(s) in {recordings_dir}")
                    if video_files:
                        log.info(f"[MetricsServer] Video files: {[f.name for f in video_files[:5]]}{'...' if len(video_files) > 5 else ''}")
                    
                    if not video_files:
                        return jsonify({
                            'success': False,
                            'message': f'No video files found in {recordings_dir}',
                            'videos_queued': 0,
                            'searched_directory': str(recordings_dir)
                        }), 200
                    
                    # Queue all videos for processing
                    queued_count = 0
                    errors = []
                    for vid_path in sorted(video_files):
                        try:
                            # Try to find corresponding GPS log
                            # Look in the same directory as the video file first, then in root
                            gps_log = None
                            video_stem = vid_path.stem
                            video_dir = vid_path.parent
                            
                            # Look for GPS log with same name (check video's directory first, then root)
                            gps_log_paths = [
                                video_dir / f"{video_stem}.json",  # Same directory as video
                                video_dir / f"{video_stem}_gps.json",  # Same directory as video
                                recordings_dir / f"{video_stem}.json",  # Root directory
                                recordings_dir / f"{video_stem}_gps.json",  # Root directory
                            ]
                            for gps_path in gps_log_paths:
                                if gps_path.exists():
                                    gps_log = str(gps_path)
                                    log.info(f"[MetricsServer] Found GPS log for {vid_path.name}: {gps_path.name}")
                                    break
                            
                            # Extract camera_id from filename (format: camera_id_timestamp_chunkXXXX)
                            parts = video_stem.split('_')
                            vid_camera_id = parts[0] if parts else camera_id
                            
                            processing_queue.queue_video_processing(
                                video_path=str(vid_path),
                                camera_id=vid_camera_id,
                                gps_log_path=gps_log,
                                detect_every_n=detect_every_n,
                                detection_mode=detection_mode
                            )
                            queued_count += 1
                            log.info(f"[MetricsServer] Queued video {queued_count}/{len(video_files)}: {vid_path.name} (camera: {vid_camera_id})")
                        except Exception as e:
                            error_msg = f"Error queueing {vid_path.name}: {str(e)}"
                            errors.append(error_msg)
                            log.info(f"[MetricsServer] {error_msg}")
                            import traceback
                            traceback.print_exc()
                    
                    if queued_count > 0 and hasattr(processing_queue, 'notify_recording_stopped'):
                        processing_queue.notify_recording_stopped()

                    if errors:
                        return jsonify({
                            'success': queued_count > 0,
                            'message': f'Queued {queued_count} video(s) for processing, {len(errors)} error(s)',
                            'videos_queued': queued_count,
                            'errors': errors
                        }), 200 if queued_count > 0 else 500
                    
                    log.info(f"[MetricsServer] Successfully queued {queued_count} video(s) for processing")
                    return jsonify({
                        'success': True,
                        'message': f'Queued {queued_count} video(s) for processing',
                        'videos_queued': queued_count,
                        'total_found': len(video_files),
                        'directory': str(recordings_dir)
                    }), 200
                
                # Process single video
                if not video_path:
                    return jsonify({'error': 'video_path is required when process_all is false'}), 400
                
                # Queue video processing
                processing_queue.queue_video_processing(
                    video_path=video_path,
                    camera_id=camera_id,
                    gps_log_path=gps_log_path,
                    detect_every_n=detect_every_n,
                    detection_mode=detection_mode
                )
                
                if hasattr(processing_queue, 'notify_recording_stopped'):
                    processing_queue.notify_recording_stopped()

                return jsonify({
                    'success': True,
                    'message': f'Video processing queued: {Path(video_path).name}',
                    'video_path': video_path
                }), 200
                
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                log.exception("ERROR in debug_start_video_processing: %s", e)
                log.info(f"[MetricsServer] Traceback:\n{error_trace}")
                return jsonify({
                    'error': str(e),
                    'traceback': error_trace
                }), 500
        
        @self.app.route('/api/debug/start-ocr-processing', methods=['POST'])
        def debug_start_ocr_processing():
            """Debug: Manually trigger OCR processing on a crops directory or all crops in out/crops."""
            try:
                data = request.get_json() or {}
                crops_dir = data.get('crops_dir')
                process_all = data.get('process_all', False)
                video_path = data.get('video_path', '')
                camera_id = data.get('camera_id', 'test-video')
                
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'processing_queue'):
                    return jsonify({'error': 'Processing queue not available'}), 500
                
                # Use or create a processing queue (OCR per video, same as application)
                if getattr(app, 'processing_queue', None):
                    existing_defer = bool(getattr(app.processing_queue, 'defer_ocr', False))
                    if existing_defer:
                        log.info("[MetricsServer] Stopping existing processing queue (defer_ocr=True) to recreate with defer_ocr=False for debug manual OCR")
                        try:
                            app.processing_queue.stop()
                        except Exception as e:
                            log.warning("[MetricsServer] Error stopping existing queue: %s", e)
                        app.processing_queue = None

                if not getattr(app, 'processing_queue', None):
                    log.info("[MetricsServer] Initializing processing queue for debug OCR")
                    if not hasattr(app, 'video_processor') or not app.video_processor:
                        return jsonify({
                            'error': 'Video processor not available',
                            'hint': 'Please start the application first to initialize the video processor.'
                        }), 500
                    
                    if getattr(app, 'ocr', None) is None:
                        if hasattr(app, '_initialize_ocr'):
                            log.info("[MetricsServer] Initializing OCR for debug processing")
                            app._initialize_ocr()
                        if not app.ocr:
                            return jsonify({
                                'error': 'OCR not available',
                                'hint': 'Please check OCR model files.'
                            }), 500

                    try:
                        from app.processing_queue import ProcessingQueueManager
                        
                        def _on_debug_ocr_complete(video_path, crops_dir, ocr_results):
                            """Store OCR results in database when debug processing completes OCR for a job."""
                            app_ref = self.frame_storage if hasattr(self, 'frame_storage') else None
                            if not app_ref or not getattr(app_ref, 'video_frame_db', None) or not ocr_results:
                                return
                            try:
                                if hasattr(app_ref, '_store_ocr_results_in_db'):
                                    app_ref._store_ocr_results_in_db(video_path, crops_dir, ocr_results, allow_missing_gps=True)
                                    log.info("[MetricsServer] Debug OCR: stored results in database for %s", Path(video_path).name)
                            except Exception as e:
                                log.exception("[MetricsServer] Debug OCR: failed to store results in database: %s", e)

                        app.processing_queue = ProcessingQueueManager(
                            video_processor=app.video_processor,
                            ocr=app.ocr,
                            preprocessor=getattr(app, 'preprocessor', None),
                            on_video_complete=None,
                            on_ocr_complete=_on_debug_ocr_complete,
                            defer_ocr=False,
                            on_video_queue_drained=None
                        )
                        log.info("[MetricsServer] Debug processing queue created for manual OCR")
                        if hasattr(app, "_sync_offline_gpu_lock_to_gate_pipeline"):
                            app._sync_offline_gpu_lock_to_gate_pipeline()
                    except Exception as e:
                        log.exception("[MetricsServer] Failed to create debug processing queue: %s", e)
                        return jsonify({
                            'error': f'Failed to initialize processing queue: {str(e)}'
                        }), 500
                else:
                    # Eagerly initialize OCR if it hasn't been loaded yet,
                    # and link it to the processing queue.
                    if getattr(app, 'ocr', None) is None:
                        if hasattr(app, '_initialize_ocr'):
                            log.info("[MetricsServer] Initializing OCR for debug processing")
                            app._initialize_ocr()
                    
                    if app.processing_queue:
                        app.processing_queue.set_ocr(app.ocr)
                        app.processing_queue.start_ocr_worker_if_deferred()
                
                # Process all crops in out/crops folder
                if process_all or not crops_dir:
                    crops_base_dir = Path("out/crops")
                    if not crops_base_dir.exists():
                        return jsonify({'error': f'Crops directory not found: {crops_base_dir}'}), 404
                    
                    # Find all crop directories (each video has its own directory)
                    crop_dirs = []
                    for camera_dir in crops_base_dir.iterdir():
                        if camera_dir.is_dir():
                            for video_dir in camera_dir.iterdir():
                                if video_dir.is_dir():
                                    # Check if directory has image files
                                    image_files = list(video_dir.glob('*.jpg')) + list(video_dir.glob('*.png'))
                                    if image_files:
                                        crop_dirs.append((str(video_dir), camera_dir.name))
                    
                    if not crop_dirs:
                        return jsonify({
                            'success': False,
                            'message': 'No crop directories found in out/crops',
                            'ocr_jobs_queued': 0
                        }), 200
                    
                    # Queue all crop directories for OCR
                    queued_count = 0
                    for crop_path, vid_camera_id in crop_dirs:
                        app.processing_queue._queue_ocr_job(
                            video_path=video_path or crop_path,
                            crops_dir=crop_path,
                            camera_id=vid_camera_id
                        )
                        queued_count += 1
                    
                    return jsonify({
                        'success': True,
                        'message': f'Queued {queued_count} crop directory(ies) for OCR processing',
                        'ocr_jobs_queued': queued_count
                    }), 200
                
                # Process single crops directory
                if not crops_dir:
                    return jsonify({'error': 'crops_dir is required when process_all is false'}), 400
                
                # Manually queue OCR job
                app.processing_queue._queue_ocr_job(
                    video_path=video_path or crops_dir,
                    crops_dir=crops_dir,
                    camera_id=camera_id
                )
                
                return jsonify({
                    'success': True,
                    'message': f'OCR processing queued: {Path(crops_dir).name}',
                    'crops_dir': crops_dir
                }), 200
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/debug/trigger-upload', methods=['POST'])
        def debug_trigger_upload():
            """Debug: Trigger Prosper upload once (same as background thread)."""
            try:
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'upload_processed_records_and_delete'):
                    return jsonify({'success': False, 'error': 'Upload not available'}), 500
                if not getattr(app, 'video_frame_db', None):
                    return jsonify({
                        'success': False,
                        'error': 'Video frame database not available',
                    }), 400
                app.upload_processed_records_and_delete()
                if hasattr(app, '_upload_status_lock'):
                    with app._upload_status_lock:
                        status = dict(getattr(app, 'upload_status', {}))
                else:
                    status = dict(getattr(app, 'upload_status', {}))
                status['thread_alive'] = (
                    getattr(app, '_upload_thread', None) is not None and app._upload_thread.is_alive()
                )
                return jsonify({
                    'success': True,
                    'message': 'Upload triggered',
                    'upload_status': status
                }), 200
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/debug/processing-queue-status', methods=['GET'])
        def debug_processing_queue_status():
            """Debug: Get processing queue status."""
            try:
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'processing_queue'):
                    return jsonify({
                        'error': 'Processing queue not available',
                        'available': False
                    }), 200
                
                if not app.processing_queue:
                    return jsonify({
                        'error': 'Processing queue not initialized',
                        'available': False
                    }), 200
                
                status = app.processing_queue.get_status()
                return jsonify({
                    'available': True,
                    'status': status
                }), 200
                
            except Exception as e:
                return jsonify({'error': str(e), 'available': False}), 500
        
        @self.app.route('/api/start-application', methods=['POST'])
        def start_application():
            """Start the automated application workflow (load assets + recording + processing)."""
            try:
                data = request.get_json() or {}
                camera_id = data.get('camera_id')
                detection_mode = (data.get('detection_mode') or 'gate').strip().lower()
                if detection_mode not in ('gate', 'stacker'):
                    detection_mode = 'gate'
                
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app:
                    return jsonify({'error': 'Application not available'}), 500
                
                app.detection_mode = detection_mode
                log.info("[MetricsServer] Start application: detection_mode=%s", detection_mode)
                if hasattr(app, "apply_live_detection_mode"):
                    swap = app.apply_live_detection_mode(detection_mode)
                    if not swap.get("success"):
                        return jsonify({
                            "success": False,
                            "message": swap.get("message", "Failed to apply live detection mode"),
                        }), 500
                
                # Check if already running
                if app.is_recording():
                    return jsonify({
                        'success': False,
                        'message': 'Application is already running'
                    }), 400
                
                # Step 1: Initialize assets (OCR, processing queue, etc.)
                log.info(f"[MetricsServer] Initializing assets for automated processing...")
                assets_result = app.initialize_assets()
                
                if not assets_result.get('success'):
                    return jsonify({
                        'success': False,
                        'message': f'Failed to initialize assets: {assets_result.get("message", "Unknown error")}',
                        'assets_loaded': assets_result.get('assets_loaded', {})
                    }), 500
                
                log.info(f"[MetricsServer] Assets initialized successfully")
                
                # Step 2: Start auto recording (this will automatically trigger processing)
                result = app.start_recording(camera_id=camera_id)
                
                if result.get('success'):
                    return jsonify({
                        'success': True,
                        'message': 'Application started successfully',
                        'camera_id': result.get('camera_id'),
                        'workflow': 'Assets loaded (no OCR yet) → Recording → Video processing per chunk → When stop: OCR once on all crops → DB/upload',
                        'assets_loaded': assets_result.get('assets_loaded', {})
                    }), 200
                else:
                    return jsonify(result), 400
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/stop-application', methods=['POST'])
        def stop_application():
            """Stop the automated application workflow."""
            try:
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app:
                    return jsonify({'error': 'Application not available'}), 500
                
                # Stop recording
                result = app.stop_recording()
                
                return jsonify({
                    'success': True,
                    'message': 'Application stopped successfully',
                    'result': result
                }), 200
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/application-status', methods=['GET'])
        def get_application_status():
            """Get comprehensive application status."""
            try:
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app:
                    return jsonify({
                        'running': False,
                        'error': 'Application not available'
                    }), 200
                
                is_recording = app.is_recording()
                
                # Get camera_id from video recorder if recording
                camera_id = None
                if hasattr(app, 'video_recorder') and app.video_recorder:
                    if app.video_recorder.is_recording() and hasattr(app.video_recorder, 'camera_id'):
                        camera_id = app.video_recorder.camera_id
                
                # Check if gracefully shutting down (recording stopped but processing ongoing)
                is_gracefully_shutting_down = False
                is_processing_ongoing = False
                if hasattr(app, 'is_gracefully_shutting_down'):
                    is_gracefully_shutting_down = app.is_gracefully_shutting_down()
                if hasattr(app, 'is_processing_ongoing'):
                    is_processing_ongoing = app.is_processing_ongoing()
                
                # Get processing queue status if available
                queue_status = None
                if hasattr(app, 'processing_queue') and app.processing_queue:
                    queue_status = app.processing_queue.get_status()
                
                # Get Prosper upload status (include thread_alive so dashboard can show background thread)
                upload_status = None
                if hasattr(app, 'upload_status') and hasattr(app, '_upload_status_lock'):
                    with app._upload_status_lock:
                        upload_status = dict(app.upload_status)
                    upload_status['thread_alive'] = (
                        getattr(app, '_upload_thread', None) is not None and app._upload_thread.is_alive()
                    )
                
                # Startup mode (backlog processing) status
                startup_backlog_status = None
                if hasattr(app, 'get_startup_backlog_status'):
                    startup_backlog_status = app.get_startup_backlog_status()
                
                return jsonify({
                    'running': is_recording,
                    'recording': is_recording,
                    'camera_id': camera_id,
                    'gracefully_shutting_down': is_gracefully_shutting_down,
                    'processing_ongoing': is_processing_ongoing,
                    'queue_status': queue_status,
                    'processing_queue_available': hasattr(app, 'processing_queue') and app.processing_queue is not None,
                    'upload_status': upload_status,
                    'startup_backlog_status': startup_backlog_status
                }), 200
                    
            except Exception as e:
                return jsonify({
                    'running': False,
                    'error': str(e)
                }), 200
        
        @self.app.route('/api/upload-status', methods=['GET'])
        def get_upload_status():
            """Get Prosper upload status (thread, config, last run, success/failed/skipped)."""
            try:
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app or not hasattr(app, 'upload_status'):
                    empty_branch = {
                        'last_result': None,
                        'last_batch_count': 0,
                        'last_deleted_count': 0,
                        'last_error': None,
                        'total_uploaded': 0,
                        'last_response_status': None,
                        'last_response_body': None,
                    }
                    return jsonify({
                        'enabled': False,
                        'thread_alive': False,
                        'config_message': 'Application or upload status not available.',
                        'is_uploading': False,
                        'last_run_at': None,
                        'yard': dict(empty_branch),
                        'gate': dict(empty_branch),
                    }), 200
                with app._upload_status_lock:
                    out = dict(app.upload_status)
                out['thread_alive'] = (
                    getattr(app, '_upload_thread', None) is not None and app._upload_thread.is_alive()
                )
                return jsonify(out), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/processed-frame/<int:frame_number>')
        def get_processed_frame(frame_number):
            """Get a specific processed frame."""
            if self.video_processor is None:
                return jsonify({'error': 'Video processor not available'}), 500
            
            frame = self.video_processor.get_frame(frame_number)
            if frame is None:
                return jsonify({'error': 'Frame not found'}), 404
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                return jsonify({'error': 'Failed to encode frame'}), 500
            
            return Response(buffer.tobytes(), mimetype='image/jpeg')
        
        @self.app.route('/api/processed-video-stream-v1')
        def stream_processed_video():
            """Stream processed video frames as MJPEG (legacy route, redirects to v2)."""
            return Response(status=204)

        @self.app.route('/api/edge-images/<path:rel_path>', methods=['GET'])
        def get_edge_image(rel_path):
            """Serve a single JPEG (or similar) by its path relative to ``out/``,
            restricted to a whitelist of subdirectories. Used by the Prosper
            image-metadata upload — Prosper POSTs ``fileUrl`` pointing here
            and then pulls the bytes itself.

            Allowed roots (must match
            ``app.prosper_image_upload.EXPOSED_IMAGE_ROOTS``):
              - ``out/crops/`` — test-mode + yard-mode crops
              - ``out/gatevision_evidence/`` — LIVE gate-pass evidence JPEGs

            Security: refuses any path that resolves outside the allowlist.
            Only serves regular files (no dirs, no symlinks-leading-elsewhere).
            """
            try:
                # Keep this list in sync with EXPOSED_IMAGE_ROOTS in
                # app/prosper_image_upload.py — the URL builder and the
                # server MUST agree or the URL won't be servable.
                exposed_roots = ("crops", "gatevision_evidence")
                out_root = Path("out").resolve()
                target = (Path("out") / rel_path).resolve()
                # Path-traversal guard: target must be inside ``out/``.
                if out_root not in target.parents and target != out_root:
                    return jsonify({'error': 'forbidden'}), 403
                # First segment after ``out/`` must be in the allowlist.
                try:
                    first_segment = target.relative_to(out_root).parts[0]
                except (IndexError, ValueError):
                    return jsonify({'error': 'forbidden'}), 403
                if first_segment not in exposed_roots:
                    return jsonify({'error': 'forbidden'}), 403
                if not target.is_file():
                    return jsonify({'error': 'not found'}), 404
                content_type, _ = mimetypes.guess_type(target.name)
                if not content_type:
                    content_type = 'application/octet-stream'
                with open(target, 'rb') as fh:
                    data = fh.read()
                return Response(data, mimetype=content_type)
            except Exception as e:
                log.exception("[MetricsServer] /api/edge-images error: %s", e)
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/dashboard/data', methods=['GET'])
        def get_dashboard_data():
            """Get dashboard data from video_frames.db (fallback: combined_results.json)."""
            date_str = request.args.get('date')
            log.info("[MetricsServer] GET /api/dashboard/data request date=%s", date_str)
            try:
                data = self._get_dashboard_data_from_db(date_str=date_str)
                if data is None:
                    data = self._get_dashboard_data_from_json(date_str=date_str)
                    log.info("[MetricsServer] GET /api/dashboard/data response source=json kpis.trailersOnYard=%s",
                             data.get('kpis', {}).get('trailersOnYard', {}).get('value'))
                else:
                    log.info("[MetricsServer] GET /api/dashboard/data response source=db kpis.trailersOnYard=%s",
                             data.get('kpis', {}).get('trailersOnYard', {}).get('value'))
                return jsonify(data)
            except Exception as e:
                log.exception("[MetricsServer] Error getting dashboard data: %s", e)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/dashboard/events', methods=['GET'])
        def get_dashboard_events():
            """Get events from video_frames.db (fallback: combined_results.json)."""
            limit = int(request.args.get('limit', 1000))
            date_str = request.args.get('date')
            log.info("[MetricsServer] GET /api/dashboard/events request limit=%s date=%s", limit, date_str)
            try:
                events = self._get_events_from_db(limit, date_str=date_str)
                if events is None:
                    events = self._get_events_from_json(limit, date_str=date_str)
                    log.info("[MetricsServer] GET /api/dashboard/events response source=json count=%s", len(events))
                else:
                    log.info("[MetricsServer] GET /api/dashboard/events response source=db count=%s", len(events))
                return jsonify({'events': events, 'count': len(events)})
            except Exception as e:
                log.exception("[MetricsServer] Error getting dashboard events: %s", e)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/cameras', methods=['GET'])
        def get_cameras():
            """Get list of cameras from cameras.yaml with status."""
            force_check = request.args.get('force', 'false').lower() == 'true'
            log.info("[MetricsServer] GET /api/cameras request force=%s", force_check)
            try:
                cameras = self._get_cameras_with_status(force_check=force_check)
                log.info("[MetricsServer] GET /api/cameras response count=%s", len(cameras))
                return jsonify({'cameras': cameras})
            except Exception as e:
                log.exception("[MetricsServer] Error getting cameras: %s", e)
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/gatevision/status', methods=['GET'])
        def get_gatevision_status():
            """Get GateVision status, gate camera list, and recent fused gate-pass events."""
            try:
                if self.frame_storage and hasattr(self.frame_storage, 'get_gatevision_status'):
                    data = self.frame_storage.get_gatevision_status()
                    return jsonify(data)
                return jsonify({'stats': {}, 'recent_events': [], 'gate_cameras': []})
            except Exception as e:
                log.exception("[MetricsServer] Error getting GateVision status: %s", e)
                return jsonify({'error': str(e), 'stats': {}, 'recent_events': [], 'gate_cameras': []}), 500

        @self.app.route('/api/gatevision/config', methods=['POST'])
        def update_gatevision_config():
            """Update GateVision runtime config values without restarting the app."""
            try:
                payload = request.get_json(silent=True) or {}
                if self.frame_storage and hasattr(self.frame_storage, 'update_gatevision_config'):
                    result = self.frame_storage.update_gatevision_config(payload)
                    code = 200 if result.get('success') else 400
                    return jsonify(result), code
                return jsonify({'success': False, 'message': 'Frame storage not available'}), 503
            except Exception as e:
                log.exception("[MetricsServer] Error updating GateVision config: %s", e)
                return jsonify({'success': False, 'message': str(e)}), 500

        @self.app.route('/api/gatevision/reload-config', methods=['POST'])
        def reload_gatevision_config():
            """Reload GateVision config from config/cameras.yaml and apply at runtime."""
            try:
                if self.frame_storage and hasattr(self.frame_storage, 'reload_gatevision_config_from_file'):
                    result = self.frame_storage.reload_gatevision_config_from_file()
                    code = 200 if result.get('success') else 400
                    return jsonify(result), code
                return jsonify({'success': False, 'message': 'Frame storage not available'}), 503
            except Exception as e:
                log.exception("[MetricsServer] Error reloading GateVision config: %s", e)
                return jsonify({'success': False, 'message': str(e)}), 500

        @self.app.route('/api/gatevision/reset', methods=['POST'])
        def reset_gatevision_state():
            """Reset GateVision runtime counters/cache/recent events."""
            try:
                if self.frame_storage and hasattr(self.frame_storage, 'reset_gatevision_runtime_state'):
                    result = self.frame_storage.reset_gatevision_runtime_state()
                    code = 200 if result.get('success') else 400
                    return jsonify(result), code
                return jsonify({'success': False, 'message': 'Frame storage not available'}), 503
            except Exception as e:
                log.exception("[MetricsServer] Error resetting GateVision state: %s", e)
                return jsonify({'success': False, 'message': str(e)}), 500

        @self.app.route('/api/gatevision/review', methods=['POST'])
        def review_gatevision_event():
            """Apply operator review decision for a fused GateVision event."""
            try:
                payload = request.get_json(silent=True) or {}
                gate_pass_id = payload.get('gate_pass_id')
                decision = payload.get('decision')
                reason = payload.get('reason', '')
                if not gate_pass_id or not decision:
                    return jsonify({'success': False, 'message': 'gate_pass_id and decision are required'}), 400
                if self.frame_storage and hasattr(self.frame_storage, 'review_gatevision_event'):
                    result = self.frame_storage.review_gatevision_event(gate_pass_id, decision, reason)
                    code = 200 if result.get('success') else 400
                    return jsonify(result), code
                return jsonify({'success': False, 'message': 'Frame storage not available'}), 503
            except Exception as e:
                log.exception("[MetricsServer] Error reviewing GateVision event: %s", e)
                return jsonify({'success': False, 'message': str(e)}), 500

        @self.app.route('/api/gatevision/test-recordings', methods=['GET'])
        @self.app.route('/api/gatevision/test-recordings/list', methods=['GET'])
        def gatevision_list_test_recordings():
            """List videos under out/recordings (or out/recording) for GateVision file testing."""
            try:
                recordings_dir = _resolve_recordings_directory()
                if not recordings_dir:
                    return jsonify(
                        {
                            'success': False,
                            'videos': [],
                            'message': 'No recordings folder found (checked out/recordings and out/recording)',
                        }
                    ), 200
                videos = []
                for f in _list_recordings_video_files(recordings_dir)[:500]:
                    try:
                        sz = f.stat().st_size if f.is_file() else 0
                    except OSError:
                        sz = 0
                    videos.append({'name': f.name, 'path': str(f), 'size': sz})
                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                default_cam = None
                if app and hasattr(app, 'get_default_gatevision_test_camera_id'):
                    default_cam = app.get_default_gatevision_test_camera_id()
                return jsonify(
                    {
                        'success': True,
                        'recordings_dir': str(recordings_dir),
                        'videos': videos,
                        'default_camera_id': default_cam or 'gatevision_test',
                    }
                ), 200
            except Exception as e:
                log.exception("[MetricsServer] gatevision_list_test_recordings: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/gatevision/camera/scan', methods=['GET', 'POST'])
        def gatevision_camera_scan():
            """Auto-discover CP Plus / ONVIF IP camera on local Ethernet."""
            try:
                from tools.rtsp_tester.scanner import scan_network_for_cameras
                cameras = scan_network_for_cameras()
                return jsonify({
                    'success': True,
                    'cameras': cameras,
                    'count': len(cameras),
                    'default_ip': cameras[0]['ip'] if cameras else '192.168.1.155'
                }), 200
            except Exception as e:
                log.exception("[MetricsServer] Camera scan error: %s", e)
                return jsonify({'success': False, 'error': str(e), 'default_ip': '192.168.1.155'}), 200

        @self.app.route('/api/gatevision/cameras', methods=['GET', 'POST'])
        def gatevision_cameras_api():
            """Get or update 4-camera Gate configuration."""
            try:
                import yaml
                from contextlib import nullcontext
                cfg_path = Path(__file__).parent.parent / "config" / "cameras.yaml"
                if request.method == 'POST':
                    body = request.get_json(silent=True) or {}
                    cams_update = body.get('cameras', [])
                    if cams_update and cfg_path.exists():
                        with open(cfg_path, 'r') as f:
                            cfg_data = yaml.safe_load(f) or {}
                        cfg_data['cameras'] = cams_update
                        with open(cfg_path, 'w') as f:
                            yaml.safe_dump(cfg_data, f, sort_keys=False)
                        return jsonify({'success': True, 'message': 'Cameras updated successfully', 'cameras': cams_update}), 200

                # GET cameras list with live status
                cameras_list = []
                app_ref = getattr(self, 'frame_storage', None)
                if cfg_path.exists():
                    with open(cfg_path, 'r') as f:
                        cfg_data = yaml.safe_load(f) or {}
                    for cam in cfg_data.get('cameras', []):
                        cid = cam.get('id')
                        active = False
                        if app_ref and hasattr(app_ref, 'latest_frames'):
                            lk = getattr(app_ref, 'frame_lock', None)
                            if lk:
                                with lk:
                                    active = cid in app_ref.latest_frames and app_ref.latest_frames[cid] is not None
                            else:
                                active = cid in app_ref.latest_frames and app_ref.latest_frames[cid] is not None
                        cameras_list.append({
                            'id': cid,
                            'name': cam.get('name', cid),
                            'rtsp_url': cam.get('rtsp_url', ''),
                            'role': cam.get('role', 'gate_front'),
                            'gate_id': cam.get('gate_id', 'gate-in'),
                            'direction': cam.get('direction', 'INBOUND'),
                            'active': active
                        })
                return jsonify({'success': True, 'cameras': cameras_list}), 200
            except Exception as e:
                log.exception("[MetricsServer] gatevision_cameras_api error: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/gatevision/camera/connect', methods=['POST'])
        def gatevision_camera_connect():
            """Dynamically connect, restart, or configure a single camera stream."""
            try:
                data = request.get_json(silent=True) or {}
                camera_id = (data.get('camera_id') or '').strip()
                rtsp_url = (data.get('rtsp_url') or '').strip()
                role = (data.get('role') or '').strip()
                gate_id = (data.get('gate_id') or '').strip()
                direction = (data.get('direction') or '').strip().upper()

                if not camera_id or not rtsp_url:
                    return jsonify({'success': False, 'error': 'camera_id and rtsp_url are required'}), 400

                app_ref = getattr(self, 'frame_storage', None)
                if app_ref and hasattr(app_ref, 'connect_live_camera'):
                    res = app_ref.connect_live_camera(
                        camera_id=camera_id,
                        rtsp_url=rtsp_url,
                        role=role,
                        gate_id=gate_id,
                        direction=direction
                    )
                    return jsonify(res), 200

                return jsonify({'success': True, 'camera_id': camera_id, 'rtsp_url': rtsp_url}), 200
            except Exception as e:
                log.exception("[MetricsServer] gatevision_camera_connect error: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/gatevision/test-recordings/run', methods=['POST'])
        def gatevision_run_test_recordings():
            """
            Queue video(s) or live RTSP streams for the GateVision pipeline:
            VideoProcessor (detect/track/crops) + batch OCR + SQLite. Crops: out/crops/<camera_id>/<video_stem>/.
            DB rows use video_path gatevision:test-<video_stem>:gate_pass (Prosper gate-events upload) and allow missing GPS.
            """
            try:
                data = request.get_json(silent=True) or {}
                process_all = bool(data.get('process_all', False))
                video_path = data.get('video_path')
                camera_id = (data.get('camera_id') or '').strip()
                detection_mode = (data.get('detection_mode') or 'gate').strip().lower()
                if detection_mode not in ('gate', 'stacker'):
                    detection_mode = 'gate'
                upload_after = bool(data.get('upload_after', False))

                app = self.frame_storage if hasattr(self, 'frame_storage') else None
                if not app:
                    return jsonify({'success': False, 'error': 'Application not available'}), 503

                _gcfg = (getattr(app, 'config', {}) or {}).get('globals', {}) or {}
                detect_every_n = int(_gcfg.get('detect_every_n', 20))
                _body_den = data.get('detect_every_n')
                if _body_den is not None and int(_body_den) != detect_every_n:
                    log.info(
                        "[MetricsServer] /api/gatevision/test-recordings/run: ignoring "
                        "body detect_every_n=%s; using cameras.yaml globals.detect_every_n=%s",
                        _body_den,
                        detect_every_n,
                    )

                if not camera_id and hasattr(app, 'get_default_gatevision_test_camera_id'):
                    camera_id = app.get_default_gatevision_test_camera_id()
                if not camera_id:
                    camera_id = 'gatevision_test'

                _batch_state = {'remaining': 0, 'lock': threading.Lock()}

                def _post_test_cleanup():
                    app_ref = self.frame_storage if hasattr(self, 'frame_storage') else None
                    if app_ref is None:
                        return
                    try:
                        log.info(
                            "[MetricsServer] Gate-test post-cleanup: unloading OCR and restoring YOLO for live mode"
                        )
                        if hasattr(app_ref, "_unload_ocr"):
                            app_ref._unload_ocr()
                        if hasattr(app_ref, "_ensure_detector_loaded"):
                            app_ref._ensure_detector_loaded()
                    except Exception as ex:
                        log.exception("[MetricsServer] Gate-test post-cleanup failed: %s", ex)

                    if not upload_after:
                        return

                    def _auto_upload():
                        try:
                            cutoff = int(os.getenv("EDGE_GATE_UPLOAD_CUTOFF_SECONDS", "5"))
                        except ValueError:
                            cutoff = 5
                        wait_s = max(1.0, float(cutoff) + 1.5)
                        log.info(
                            "[MetricsServer] Gate-test upload_after=true: waiting %.1fs for "
                            "gate-event age cutoff before chaining Prosper upload",
                            wait_s,
                        )
                        time.sleep(wait_s)
                        try:
                            if hasattr(app_ref, "upload_processed_records_and_delete"):
                                log.info(
                                    "[MetricsServer] Gate-test upload_after=true: triggering Prosper upload..."
                                )
                                app_ref.upload_processed_records_and_delete()
                                log.info("[MetricsServer] Gate-test upload_after=true: upload chain complete")
                        except Exception as ex:
                            log.exception(
                                "[MetricsServer] Gate-test upload_after=true: upload chain failed: %s",
                                ex,
                            )

                    threading.Thread(
                        target=_auto_upload,
                        name="GateTestAutoUpload",
                        daemon=True,
                    ).start()

                def _gate_test_ocr_cb(video_path_inner: str, crops_dir: str, ocr_results):
                    app_ref = self.frame_storage if hasattr(self, 'frame_storage') else None
                    try:
                        if app_ref and hasattr(app_ref, 'record_gatevision_test_ocr_results'):
                            app_ref.record_gatevision_test_ocr_results(video_path_inner, crops_dir, ocr_results)
                    except Exception as ex:
                        log.exception("[MetricsServer] _gate_test_ocr_cb failed: %s", ex)
                    # Decrement remaining counter and run post-test cleanup if
                    # this was the last OCR job in the batch.
                    is_last = False
                    with _batch_state['lock']:
                        _batch_state['remaining'] -= 1
                        is_last = _batch_state['remaining'] <= 0
                    if is_last:
                        _post_test_cleanup()

                def _drain_to_ocr_phase(pending_jobs):
                    app_ref = self.frame_storage if hasattr(self, 'frame_storage') else None
                    if app_ref is None:
                        log.error("[MetricsServer] _drain_to_ocr_phase: app missing; cannot run OCR")
                        return
                    try:
                        log.info(
                            "[MetricsServer] Gate-test transition: video phase done, swapping YOLO→OCR for %s job(s)",
                            len(pending_jobs),
                        )
                        if hasattr(app_ref, "_unload_detector"):
                            app_ref._unload_detector()
                        if not getattr(app_ref, "ocr", None) and hasattr(app_ref, "_initialize_ocr"):
                            app_ref._initialize_ocr()
                        if not getattr(app_ref, "ocr", None):
                            log.error(
                                "[MetricsServer] _drain_to_ocr_phase: OCR failed to load; %s job(s) abandoned",
                                len(pending_jobs),
                            )
                            _post_test_cleanup()
                            return
                        pq_ref = getattr(app_ref, "processing_queue", None)
                        if pq_ref is None:
                            log.error("[MetricsServer] _drain_to_ocr_phase: processing_queue missing")
                            _post_test_cleanup()
                            return
                        pq_ref.set_ocr(app_ref.ocr)
                        annotated_jobs = []
                        for j in pending_jobs:
                            j2 = dict(j)
                            j2['on_ocr_complete'] = _gate_test_ocr_cb
                            annotated_jobs.append(j2)
                        with _batch_state['lock']:
                            _batch_state['remaining'] = len(annotated_jobs)
                        pq_ref.queue_ocr_jobs(annotated_jobs)
                        pq_ref.start_ocr_worker_if_deferred()
                    except Exception as ex:
                        log.exception("[MetricsServer] _drain_to_ocr_phase failed: %s", ex)
                        _post_test_cleanup()

                if getattr(app, 'ocr', None) is not None and hasattr(app, '_unload_ocr'):
                    app._unload_ocr()
                if hasattr(app, "_ensure_detector_loaded"):
                    if not app._ensure_detector_loaded():
                        return jsonify({
                            'success': False,
                            'error': 'Failed to load YOLO detector',
                            'hint': 'Restart the application; build_live_detector returned None.',
                        }), 503

                pq, err = _ensure_file_video_processing_queue(
                    app, defer_ocr=True, on_video_queue_drained=_drain_to_ocr_phase,
                )
                if err:
                    return err[0], err[1]
                if hasattr(pq, 'notify_recording_started'):
                    pq.notify_recording_started()

                recordings_dir = _resolve_recordings_directory()

                if process_all:
                    if not recordings_dir:
                        return jsonify(
                            {
                                'success': False,
                                'error': 'Recordings directory not found',
                                'checked_paths': ['out/recordings', 'out/recording'],
                            }
                        ), 404
                    video_files = _list_recordings_video_files(recordings_dir)
                    if not video_files:
                        return jsonify(
                            {
                                'success': False,
                                'videos_queued': 0,
                                'message': f'No video files in {recordings_dir}',
                            }
                        ), 200
                    queued = 0
                    errors = []
                    for vid_path in video_files:
                        try:
                            video_stem = vid_path.stem
                            gps_log = None
                            for gps_path in (
                                vid_path.parent / f"{video_stem}.json",
                                vid_path.parent / f"{video_stem}_gps.json",
                                recordings_dir / f"{video_stem}.json",
                                recordings_dir / f"{video_stem}_gps.json",
                            ):
                                if gps_path.exists():
                                    gps_log = str(gps_path)
                                    break
                            pq.queue_video_processing(
                                video_path=str(vid_path),
                                camera_id=camera_id,
                                gps_log_path=gps_log,
                                detect_every_n=detect_every_n,
                                detection_mode=detection_mode,
                            )
                            queued += 1
                        except Exception as e:
                            errors.append(f"{vid_path.name}: {e}")
                    if queued > 0 and hasattr(pq, 'notify_recording_stopped'):
                        pq.notify_recording_stopped()
                    return jsonify(
                        {
                            'success': queued > 0,
                            'videos_queued': queued,
                            'total_found': len(video_files),
                            'camera_id': camera_id,
                            'errors': errors or None,
                            'message': f'Queued {queued} video(s) for GateVision-style offline processing',
                        }
                    ), 200 if queued > 0 else 500

                if not process_all and not video_path:
                    return jsonify(
                        {'success': False, 'error': 'video_path is required unless process_all is true'}
                    ), 400

                is_rtsp_stream = str(video_path).strip().lower().startswith("rtsp://")

                if not is_rtsp_stream and not Path(video_path).is_file():
                    return jsonify({'success': False, 'error': f'Video not found: {video_path}'}), 404

                if is_rtsp_stream:
                    vp_str = str(video_path).strip()
                    video_stem = f"live_camera_{int(time.time())}"
                    gps_log = None
                else:
                    vp = Path(video_path)
                    vp_str = str(vp.resolve())
                    video_stem = vp.stem
                    gps_log = None
                    if recordings_dir:
                        for gps_path in (
                            vp.parent / f"{video_stem}.json",
                            vp.parent / f"{video_stem}_gps.json",
                            recordings_dir / f"{video_stem}.json",
                            recordings_dir / f"{video_stem}_gps.json",
                        ):
                            if gps_path.exists():
                                gps_log = str(gps_path)
                                break

                pq.queue_video_processing(
                    video_path=vp_str,
                    camera_id=camera_id,
                    gps_log_path=gps_log,
                    detect_every_n=detect_every_n,
                    detection_mode=detection_mode,
                )
                if hasattr(pq, 'notify_recording_stopped'):
                    pq.notify_recording_stopped()
                return jsonify(
                    {
                        'success': True,
                        'message': (
                            f'Queued {"Live Camera RTSP Stream" if is_rtsp_stream else Path(video_path).name} '
                            'for Gate Vision processing'
                        ),
                        'video_path': vp_str,
                        'camera_id': camera_id,
                        'crops_hint': f'out/crops/{camera_id}/{video_stem}/',
                    }
                ), 200
            except Exception as e:
                log.exception("[MetricsServer] gatevision_run_test_recordings: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/api/live-detection-mode', methods=['POST'])
        def set_live_detection_mode():
            """Switch car, trailer, or stacker mode for live YardVision + GateVision."""
            try:
                payload = request.get_json(silent=True) or {}
                mode = (payload.get('detection_mode') or 'stacker').strip().lower()
                if mode not in ('car', 'trailer', 'stacker'):
                    return jsonify({'success': False, 'message': 'detection_mode must be car, trailer, or stacker'}), 400

                # 1. Update globals_cfg (used by process_video and other routes)
                try:
                    from app.config import globals_cfg
                    globals_cfg['detection_mode'] = mode
                except Exception:
                    pass

                # 2. Update the app instance attribute directly so get_gatevision_status() reflects the new mode
                app_ref = self.frame_storage if hasattr(self, 'frame_storage') else None
                if app_ref:
                    app_ref.detection_mode = mode
                    log.info(f"[MetricsServer] Updated app.detection_mode -> {mode}")

                    # 3. If there's a deeper apply method (car/trailer detector swap), call it
                    if hasattr(app_ref, 'apply_live_detection_mode') and mode in ('car', 'trailer'):
                        try:
                            result = app_ref.apply_live_detection_mode(mode)
                            code = 200 if result.get('success') else 500
                            return jsonify(result), code
                        except Exception as apply_err:
                            log.warning("[MetricsServer] apply_live_detection_mode failed: %s", apply_err)

                log.info(f"[MetricsServer] Live detection mode set to: {mode}")
                return jsonify({'success': True, 'detection_mode': mode}), 200
            except Exception as e:
                log.exception("[MetricsServer] Error setting live detection mode: %s", e)
                return jsonify({'success': False, 'message': str(e)}), 500
        
        @self.app.route('/api/inventory', methods=['GET'])
        def get_inventory():
            """Get inventory data from video_frames.db (fallback: combined_results.json)."""
            log.info("[MetricsServer] GET /api/inventory request")
            try:
                data = self._get_inventory_from_db()
                if data is None:
                    data = self._get_inventory_from_json()
                    log.info("[MetricsServer] GET /api/inventory response source=json trailers=%s",
                             len(data.get('trailers', [])))
                else:
                    log.info("[MetricsServer] GET /api/inventory response source=db trailers=%s",
                             len(data.get('trailers', [])))
                return jsonify(data)
            except Exception as e:
                log.exception("[MetricsServer] Error getting inventory data: %s", e)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/yard-view', methods=['GET'])
        def get_yard_view():
            """Get yard view (spots/lanes) from video_frames.db."""
            log.info("[MetricsServer] GET /api/yard-view request")
            try:
                data = self._get_yard_view_from_db()
                log.info("[MetricsServer] GET /api/yard-view response spots=%s lanes=%s",
                         len(data.get('spots', [])), len(data.get('lanes', [])))
                return jsonify(data)
            except Exception as e:
                log.exception("[MetricsServer] Error getting yard view: %s", e)
                return jsonify({'error': str(e), 'spots': [], 'lanes': []}), 500
        
        @self.app.route('/api/reports', methods=['GET'])
        def get_reports():
            """Get reports (daily/weekly/monthly) from video_frames.db."""
            log.info("[MetricsServer] GET /api/reports request")
            try:
                data = self._get_reports_from_db()
                log.info("[MetricsServer] GET /api/reports response ok")
                return jsonify(data)
            except Exception as e:
                log.exception("[MetricsServer] Error getting reports: %s", e)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/data-processor/status', methods=['GET'])
        def data_processor_status():
            """Get data processor status (running, spots loaded)."""
            try:
                svc = self.get_data_processor_service()
                stats = svc.get_statistics()
                return jsonify({
                    'running': svc.running,
                    'parking_spots_loaded': len(svc.parking_spots),
                    **stats
                }), 200
            except Exception as e:
                log.exception("Data processor status error: %s", e)
                return jsonify({'error': str(e), 'running': False, 'parking_spots_loaded': 0}), 500
        
        @self.app.route('/api/data-processor/load-csv', methods=['POST'])
        def data_processor_load_csv():
            """Upload a CSV file and load reference parking spots; then run one processing cycle."""
            try:
                if 'file' not in request.files and 'csv' not in request.files:
                    return jsonify({'success': False, 'error': 'No file part; use form field "file" or "csv"'}), 400
                f = request.files.get('file') or request.files.get('csv')
                if not f or f.filename == '':
                    return jsonify({'success': False, 'error': 'No file selected'}), 400
                if not (f.filename or '').lower().endswith('.csv'):
                    return jsonify({'success': False, 'error': 'File must be a CSV'}), 400
                filename = secure_filename(f.filename) or 'spots.csv'
                csv_path = self.upload_dir / f"data_processor_{uuid.uuid4().hex}_{filename}"
                f.save(str(csv_path))
                try:
                    svc = self.get_data_processor_service()
                    svc.load_parking_spots_from_csv(str(csv_path))
                    spots_count = len(svc.parking_spots)
                    if spots_count == 0:
                        return jsonify({
                            'success': False,
                            'error': 'No valid parking spots found in CSV. Expected columns: id, name, latitude, longitude (or lat, lon)'
                        }), 200
                    result = svc.run_all()
                    processed = result.get('processed', 0)
                    log.info(
                        "Data processor load-csv: loaded %d spots, processed %d record(s).",
                        spots_count, processed
                    )
                    return jsonify({
                        'success': True,
                        'parking_spots_loaded': spots_count,
                        'processed': processed,
                        'message': f'Loaded {spots_count} spots and processed {processed} record(s).'
                    }), 200
                finally:
                    if csv_path.exists():
                        try:
                            csv_path.unlink()
                        except OSError:
                            pass
            except Exception as e:
                log.exception("Data processor load CSV error: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/data-processor/run', methods=['POST'])
        def data_processor_run():
            """Run one processing cycle with current parking spots."""
            try:
                svc = self.get_data_processor_service()
                if not svc.parking_spots:
                    return jsonify({'success': False, 'error': 'No parking spots loaded. Load a CSV first.'}), 200
                result = svc.run_once()
                return jsonify({
                    'success': result.get('success', True),
                    'processed': result.get('processed', 0),
                    'message': f"Processed {result.get('processed', 0)} record(s)."
                }), 200
            except Exception as e:
                log.exception("Data processor run error: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/data-processor/start', methods=['POST'])
        def data_processor_start():
            """Start the data processor service (runs periodically; loads spots from all config/*.csv each cycle)."""
            try:
                svc = self.get_data_processor_service()
                svc.start()
                return jsonify({'success': True, 'message': 'Data processor started.'}), 200
            except Exception as e:
                log.exception("Data processor start error: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/data-processor/stop', methods=['POST'])
        def data_processor_stop():
            """Stop the data processor service."""
            try:
                if self._data_processor_service is None:
                    return jsonify({'success': True, 'message': 'Data processor was not running.'}), 200
                self._data_processor_service.stop()
                return jsonify({'success': True, 'message': 'Data processor stopped.'}), 200
            except Exception as e:
                log.exception("Data processor stop error: %s", e)
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/api/video-frame-records', methods=['GET'])
        def get_video_frame_records():
            """Get video frame records from video_frames.db."""
            limit = request.args.get('limit', default=50, type=int)
            offset = request.args.get('offset', default=0, type=int)
            is_processed = request.args.get('is_processed', default=None, type=str)
            camera_id = request.args.get('camera_id', default=None, type=str)
            log.info("[MetricsServer] GET /api/video-frame-records limit=%s offset=%s is_processed=%s camera_id=%s",
                     limit, offset, is_processed, camera_id)
            try:
                from app.video_frame_db import VideoFrameDB
                db = VideoFrameDB(db_path="data/video_frames.db")
                is_processed_bool = None
                if is_processed is not None:
                    is_processed_bool = is_processed.lower() == 'true'
                records = db.get_all_records(
                    limit=limit,
                    offset=offset,
                    is_processed=is_processed_bool,
                    camera_id=camera_id if camera_id else None
                )
                stats = db.get_statistics()
                log.info("[MetricsServer] GET /api/video-frame-records response records=%s total=%s",
                         len(records), stats.get('total', 0))
                return jsonify({
                    'records': records,
                    'stats': stats,
                    'limit': limit,
                    'offset': offset,
                    'total': stats.get('total', 0)
                })
            except Exception as e:
                log.exception("[MetricsServer] Error getting video frame records: %s", e)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/gatevision-records', methods=['GET'])
        def get_gatevision_records():
            """Browse gatevision_records (live + test gate events).

            Query params: limit, offset, source ('live' | 'test' | omit for both).
            Sibling of /api/video-frame-records — that one returns yardvision rows only,
            this one returns gatevision rows only, after the YardVision/GateVision split.
            """
            limit = request.args.get('limit', default=50, type=int)
            offset = request.args.get('offset', default=0, type=int)
            source = request.args.get('source', default=None, type=str)
            try:
                from app.video_frame_db import VideoFrameDB
                db = VideoFrameDB(db_path="data/video_frames.db")
                records = db.get_all_gatevision_records(
                    limit=limit, offset=offset,
                    source=(source if source in ('live', 'test') else None),
                )
                stats = db.get_statistics()
                return jsonify({
                    'records': records,
                    'stats': stats,
                    'limit': limit,
                    'offset': offset,
                    'gate_total': stats.get('gate_total', 0),
                })
            except Exception as e:
                log.exception("[MetricsServer] Error getting gatevision records: %s", e)
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/processed-video-frame.jpg', methods=['GET'])
        def processed_video_frame_jpg():
            """Return the latest annotated frame as a single JPEG image."""
            try:
                frame = None
                vp = getattr(self, 'video_processor', None)
                if vp is not None:
                    if hasattr(vp, 'last_annotated_frame') and vp.last_annotated_frame is not None:
                        with vp.lock:
                            frame = vp.last_annotated_frame.copy()
                    elif hasattr(vp, 'processed_frames') and vp.processed_frames:
                        with vp.lock:
                            keys = list(vp.processed_frames.keys())
                            if keys:
                                frame = vp.processed_frames[keys[-1]].copy()

                if frame is None:
                    app_ref = getattr(self, 'frame_storage', None)
                    if app_ref is not None:
                        sp = getattr(app_ref, 'stacker_pipeline', None)
                        if sp is not None:
                            f = getattr(sp, 'last_annotated_frame', None)
                            if f is not None:
                                frame = f.copy()

                if frame is not None:
                    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        return Response(jpeg.tobytes(), mimetype='image/jpeg', headers={'Cache-Control': 'no-store, no-cache'})
                
                placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Gate Vision AI - Ready", (170, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                ret, jpeg = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 70])
                return Response(jpeg.tobytes(), mimetype='image/jpeg', headers={'Cache-Control': 'no-store, no-cache'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/processed-video-stream', methods=['GET'])
        def processed_video_stream():
            """Stream real-time annotated video frames (Gate HUD) as MJPEG."""
            def generate_mjpeg():
                import time as _time
                while True:
                    frame = None
                    try:
                        vp = getattr(self, 'video_processor', None)
                        if vp is None:
                            app_ref = getattr(self, 'frame_storage', None)
                            if app_ref is not None:
                                vp = getattr(app_ref, 'video_processor', None)

                        if vp is not None:
                            if hasattr(vp, 'last_annotated_frame') and vp.last_annotated_frame is not None:
                                with vp.lock:
                                    frame = vp.last_annotated_frame.copy()
                            elif hasattr(vp, 'processed_frames') and vp.processed_frames:
                                with vp.lock:
                                    keys = list(vp.processed_frames.keys())
                                    if keys:
                                        frame = vp.processed_frames[keys[-1]].copy()

                        if frame is None:
                            app_ref = getattr(self, 'frame_storage', None)
                            if app_ref is not None:
                                # Check live gate camera frame storage
                                if hasattr(app_ref, 'get_latest_frame'):
                                    for cam_id in ['gate-camera-01', 'gate_front', 'gatevision_test']:
                                        f = app_ref.get_latest_frame(cam_id)
                                        if f is not None:
                                            frame = f.copy()
                                            break

                        if frame is not None:
                            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            if ret:
                                b_data = jpeg.tobytes()
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n'
                                       b'Content-Length: ' + str(len(b_data)).encode() + b'\r\n\r\n' +
                                       b_data + b'\r\n')
                        else:
                            placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                            cv2.putText(placeholder, "Gate Vision AI - Ready to Stream", (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
                            ret, jpeg = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            if ret:
                                b_data = jpeg.tobytes()
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n'
                                       b'Content-Length: ' + str(len(b_data)).encode() + b'\r\n\r\n' +
                                       b_data + b'\r\n')
                    except Exception as e:
                        log.debug(f"[MJPEG] Frame error: {e}")

                    _time.sleep(0.033)  # ~30 fps cap

            res = Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')
            res.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            res.headers['Pragma'] = 'no-cache'
            res.headers['Expires'] = '0'
            return res

        @self.app.route('/video_feed', methods=['GET'])
        @self.app.route('/api/camera-feed', methods=['GET'])
        @self.app.route('/api/camera-feed/<camera_id>', methods=['GET'])
        @self.app.route('/stream/<camera_id>', methods=['GET'])
        def live_camera_stream_route(camera_id=None):
            """Stream live camera feed as MJPEG."""
            def generate_live_mjpeg():
                import time as _time
                while True:
                    frame = None
                    try:
                        app_ref = getattr(self, 'frame_storage', None)
                        if app_ref is not None:
                            # 1. Try get_latest_frame
                            if hasattr(app_ref, 'get_latest_frame'):
                                frame = app_ref.get_latest_frame(camera_id)
                            # 2. Try latest_frames directly
                            if frame is None and hasattr(app_ref, 'latest_frames'):
                                lk = getattr(app_ref, 'frame_lock', None)
                                if lk:
                                    with lk:
                                        if camera_id and camera_id in app_ref.latest_frames:
                                            f = app_ref.latest_frames[camera_id]
                                            if f is not None:
                                                frame = f.copy()
                                        elif not camera_id and app_ref.latest_frames:
                                            for cid, f in app_ref.latest_frames.items():
                                                if f is not None:
                                                    frame = f.copy()
                                                    break
                            # 3. Fallback to video_processor only if no camera_id specified
                            if frame is None and not camera_id:
                                vp = getattr(self, 'video_processor', None) or getattr(app_ref, 'video_processor', None)
                                if vp and hasattr(vp, 'last_annotated_frame') and vp.last_annotated_frame is not None:
                                    with vp.lock:
                                        frame = vp.last_annotated_frame.copy()

                        if frame is not None:
                            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            if ret:
                                b_data = jpeg.tobytes()
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n'
                                       b'Content-Length: ' + str(len(b_data)).encode() + b'\r\n\r\n' +
                                       b_data + b'\r\n')
                        else:
                            placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
                            cv2.putText(placeholder, "Live Camera - Connecting...", (140, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
                            ret, jpeg = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            if ret:
                                b_data = jpeg.tobytes()
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n'
                                       b'Content-Length: ' + str(len(b_data)).encode() + b'\r\n\r\n' +
                                       b_data + b'\r\n')
                    except Exception as e:
                        log.debug(f"[LiveMJPEG] Frame error: {e}")
                    _time.sleep(0.033)

            res = Response(generate_live_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')
            res.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            res.headers['Pragma'] = 'no-cache'
            res.headers['Expires'] = '0'
            return res

        # ---------------------------------------------------------------- Database Maintenance API

        @self.app.route('/api/debug/clear-database', methods=['POST', 'GET'])
        def debug_clear_database():
            """Clear all records from video_frames.db tables for clean testing."""
            try:
                import sqlite3
                db_path = Path(__file__).parent.parent / "data" / "video_frames.db"
                if db_path.exists():
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = [row[0] for row in cursor.fetchall()]
                    cleared = []
                    for table in tables:
                        if table != "sqlite_sequence":
                            cursor.execute(f"DELETE FROM {table};")
                            cleared.append(table)
                    conn.commit()
                    conn.execute("VACUUM;")
                    conn.close()
                    log.info("[MetricsServer] Cleared all database tables: %s", cleared)
                    return jsonify({'success': True, 'cleared_tables': cleared})
                return jsonify({'success': False, 'message': 'DB file not found'}), 404
            except Exception as e:
                log.exception('[MetricsServer] debug_clear_database error: %s', e)
                return jsonify({'success': False, 'error': str(e)}), 500


        # Static file routes must be LAST (after all API routes)
        # to avoid catching API requests


        @self.app.route('/out/<path:path>')
        def serve_out_files(path):
            """Serve files from out/ directory (stacker_crops, crops, etc.)."""
            try:
                out_dir = Path(__file__).parent.parent / 'out'
                target_path = out_dir / path
                if not target_path.exists() or not target_path.is_file():
                    from werkzeug.exceptions import NotFound
                    raise NotFound()
                return send_from_directory(str(target_path.parent), target_path.name)
            except Exception as e:
                log.exception('[MetricsServer] serve_out_files error: %s', e)
                return jsonify({'error': str(e)}), 404

        @self.app.route('/')
        def index():
            """Serve dashboard index.html."""
            web_dir = Path(__file__).parent.parent / 'web'
            return send_from_directory(str(web_dir), 'index.html')
        
        @self.app.route('/<path:path>')
        def static_files(path):
            """Serve static files (JS, CSS). Excludes /api/* paths."""
            # Don't serve API paths as static files
            if path.startswith('api/'):
                return jsonify({'error': 'API endpoint not found'}), 404
            
            web_dir = Path(__file__).parent.parent / 'web'
            file_path = web_dir / path
            
            # Check if file exists before trying to serve it
            if not file_path.exists() or not file_path.is_file():
                # File doesn't exist - return 404 (this will be handled by NotFound handler)
                from werkzeug.exceptions import NotFound
                raise NotFound()
            
            # File exists - serve it
            return send_from_directory(str(web_dir), path)
    
    def _get_video_frame_db(self):
        """Get VideoFrameDB instance (uses data/video_frames.db)."""
        from app.video_frame_db import VideoFrameDB
        db_path = Path(__file__).parent.parent / "data" / "video_frames.db"
        return VideoFrameDB(db_path=str(db_path))
    
    def _filter_records_by_date(self, records: List[Dict], date_str: Optional[str]) -> List[Dict]:
        """Filter records by created_on date (YYYY-MM-DD). If date_str is None, use today."""
        from datetime import date as date_type
        if not date_str:
            date_str = datetime.utcnow().strftime('%Y-%m-%d')
        try:
            filter_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            return records
        out = []
        for r in records:
            created = r.get('created_on') or r.get('timestamp') or ''
            if not created:
                continue
            try:
                if 'T' in str(created):
                    rec_date = datetime.fromisoformat(str(created).replace('Z', '+00:00')).date()
                else:
                    rec_date = datetime.strptime(str(created)[:10], '%Y-%m-%d').date()
                if rec_date == filter_date:
                    out.append(r)
            except (ValueError, TypeError):
                continue
        return out
    
    def _get_dashboard_data_from_db(self, date_str: str = None) -> Optional[Dict]:
        """Build dashboard KPIs/charts from video_frames.db. Returns None if DB has no data."""
        try:
            db = self._get_video_frame_db()
            stats = db.get_statistics()
            if stats.get('total', 0) == 0:
                return None
            records = db.get_all_records(limit=2000, offset=0, is_processed=None, camera_id=None)
            records = self._filter_records_by_date(records, date_str)
            if not records and date_str:
                return None
            if not records:
                records = db.get_all_records(limit=500, offset=0, is_processed=None, camera_id=None)
            from collections import defaultdict
            unique_tracks = set(r.get('track_id') for r in records if r.get('track_id') is not None)
            unique_spots = set()
            for r in records:
                s = (r.get('assigned_spot_name') or r.get('processed_comment') or '').strip()
                if s and s.lower() != 'unknown':
                    unique_spots.add(s)
            total_detections = len(records)
            ocr_ok = sum(1 for r in records if (r.get('licence_plate_trailer') or '').strip())
            ocr_accuracy = (ocr_ok / total_detections * 100) if total_detections else 0
            low_conf = sum(1 for r in records if (r.get('confidence') or 0) < 0.5)
            cameras = self._get_cameras_with_status()
            online = sum(1 for c in cameras if c.get('status') == 'online')
            degraded = sum(1 for c in cameras if c.get('status') == 'degraded')
            hourly = defaultdict(lambda: {'total': 0, 'total_det': 0.0, 'total_ocr': 0.0, 'ocr_n': 0})
            for r in records:
                created = r.get('created_on') or r.get('timestamp') or ''
                if created:
                    try:
                        dt = datetime.fromisoformat(str(created).replace('Z', '+00:00'))
                        key = dt.strftime('%H:00')
                        hourly[key]['total'] += 1
                        conf = float(r.get('confidence') or 0)
                        hourly[key]['total_det'] += conf
                        if (r.get('licence_plate_trailer') or '').strip():
                            hourly[key]['total_ocr'] += conf
                            hourly[key]['ocr_n'] += 1
                    except (ValueError, TypeError):
                        pass
            accuracy_chart = []
            for h in range(24):
                key = f'{h:02d}:00'
                d = hourly.get(key, {'total': 0, 'total_det': 0.0, 'total_ocr': 0.0, 'ocr_n': 0})
                det_pct = (d['total_det'] / d['total'] * 100) if d['total'] else 0
                ocr_pct = (d['total_ocr'] / d['ocr_n'] * 100) if d['ocr_n'] else 0
                accuracy_chart.append({'time': key, 'detection': round(det_pct, 1), 'ocr': round(ocr_pct, 1)})
            spot_counts = defaultdict(int)
            for r in records:
                s = (r.get('assigned_spot_name') or '').strip() or (r.get('processed_comment') or '').strip()
                if s and s.lower() != 'unknown':
                    spot_counts[s] += 1
            yard_util = [{'lane': k, 'utilization': v} for k, v in sorted(spot_counts.items())]
            return {
                'kpis': {
                    'trailersOnYard': {'value': len(unique_tracks), 'change': f'+{len(unique_tracks)}', 'icon': '🚛'},
                    'newDetections24h': {'value': total_detections, 'ocrAccuracy': f'{ocr_accuracy:.1f}%', 'icon': '📈'},
                    'anomalies': {'value': low_conf, 'description': f'{low_conf} low confidence', 'icon': '⚠️'},
                    'camerasOnline': {'value': online, 'degraded': degraded, 'icon': '📷'},
                },
                'queueStatus': {'ingestQ': 0, 'ocrQ': 0, 'pubQ': 0},
                'accuracyChart': accuracy_chart,
                'yardUtilization': yard_util,
                'cameraHealth': cameras,
            }
        except Exception as e:
            log.debug("[MetricsServer] _get_dashboard_data_from_db failed: %s", e)
            return None
    
    def _get_events_from_db(self, limit: int, date_str: str = None) -> Optional[List[Dict]]:
        """Get events from video_frames.db. Returns None on error or empty DB."""
        try:
            db = self._get_video_frame_db()
            stats = db.get_statistics()
            if stats.get('total', 0) == 0:
                return []
            records = db.get_all_records(limit=min(limit, 2000), offset=0, is_processed=None, camera_id=None)
            records = self._filter_records_by_date(records, date_str)
            events = []
            for r in records:
                ts = r.get('timestamp') or r.get('created_on') or ''
                events.append({
                    'ts_iso': ts,
                    'camera_id': r.get('camera_id') or 'N/A',
                    'track_id': r.get('track_id') if r.get('track_id') is not None else 'N/A',
                    'text': (r.get('licence_plate_trailer') or '').strip() or '',
                    'conf': float(r.get('confidence') or 0),
                    'spot': (r.get('assigned_spot_name') or '').strip() or 'unknown',
                    'ocr_conf': float(r.get('confidence') or 0),
                    'lat': r.get('latitude'),
                    'lon': r.get('longitude'),
                })
            events.sort(key=lambda x: x.get('ts_iso', ''), reverse=True)
            return events[:limit]
        except Exception as e:
            log.debug("[MetricsServer] _get_events_from_db failed: %s", e)
            return None
    
    def _get_inventory_from_db(self) -> Optional[Dict]:
        """Get inventory (trailers + stats) from video_frames.db. Returns None on error."""
        from collections import defaultdict
        try:
            db = self._get_video_frame_db()
            stats = db.get_statistics()
            if stats.get('total', 0) == 0:
                return None
            records = db.get_all_records(limit=1000, offset=0, is_processed=None, camera_id=None)
            by_track = defaultdict(list)
            for r in records:
                tid = r.get('track_id')
                if tid is not None:
                    by_track[tid].append(r)
            trailers = []
            for track_id, rlist in by_track.items():
                r = max(rlist, key=lambda x: (x.get('created_on') or x.get('timestamp') or ''))
                spot = (r.get('assigned_spot_name') or '').strip() or 'N/A'
                status = 'Parked' if r.get('is_processed') or (spot and spot != 'N/A') else 'In Transit'
                trailers.append({
                    'id': f"T{track_id}" if track_id is not None else f"R{r.get('id', 0)}",
                    'plate': (r.get('licence_plate_trailer') or '').strip() or 'N/A',
                    'spot': spot if spot != 'N/A' else 'N/A',
                    'status': status,
                    'detectedAt': r.get('created_on') or r.get('timestamp') or '',
                    'ocrConfidence': float(r.get('confidence') or 0),
                    'lat': r.get('latitude'),
                    'lon': r.get('longitude'),
                })
            total = len(trailers)
            parked = sum(1 for t in trailers if t['status'] == 'Parked')
            anomalies = sum(1 for rec in records if (rec.get('confidence') or 0) < 0.5)
            return {
                'trailers': trailers,
                'stats': {'total': total, 'parked': parked, 'inTransit': total - parked, 'anomalies': anomalies},
            }
        except Exception as e:
            log.debug("[MetricsServer] _get_inventory_from_db failed: %s", e)
            return None
    
    def _get_yard_view_from_db(self) -> Dict:
        """Get yard view spots/lanes from video_frames.db (assigned spots + occupancy)."""
        try:
            db = self._get_video_frame_db()
            records = db.get_all_records(limit=500, offset=0, is_processed=None, camera_id=None)
            spot_to_latest = {}
            for r in records:
                sid = (r.get('assigned_spot_id') or '').strip() or (r.get('assigned_spot_name') or '').strip()
                if not sid:
                    continue
                created = r.get('created_on') or r.get('timestamp') or ''
                if sid not in spot_to_latest or (spot_to_latest[sid].get('created_on') or '') < created:
                    spot_to_latest[sid] = r
            spots = []
            for sid, r in spot_to_latest.items():
                name = (r.get('assigned_spot_name') or sid).strip()
                lane = name.split('-')[0] if name else 'A'
                spots.append({
                    'id': sid,
                    'lane': lane,
                    'row': 1,
                    'occupied': True,
                    'trailerId': f"T{r.get('track_id')}" if r.get('track_id') is not None else None,
                    'plate': (r.get('licence_plate_trailer') or '').strip() or None,
                })
            lanes = sorted(set(s['lane'] for s in spots)) if spots else ['A', 'B', 'C', 'D', 'Dock']
            return {'spots': spots, 'lanes': lanes}
        except Exception as e:
            log.debug("[MetricsServer] _get_yard_view_from_db failed: %s", e)
            return {'spots': [], 'lanes': ['A', 'B', 'C', 'D', 'Dock']}
    
    def _get_reports_from_db(self) -> Dict:
        """Get report stats (daily/weekly/monthly) from video_frames.db."""
        try:
            db = self._get_video_frame_db()
            stats = db.get_statistics()
            all_records = db.get_all_records(limit=5000, offset=0, is_processed=None, camera_id=None)
            total = len(all_records)
            ocr_ok = sum(1 for r in all_records if (r.get('licence_plate_trailer') or '').strip())
            ocr_pct = (ocr_ok / total * 100) if total else 0
            anomalies = sum(1 for r in all_records if (r.get('confidence') or 0) < 0.5)
            today = datetime.utcnow().strftime('%Y-%m-%d')
            week_start = (datetime.utcnow() - __import__('datetime').timedelta(days=7)).strftime('%Y-%m-%d')
            month_start = (datetime.utcnow() - __import__('datetime').timedelta(days=30)).strftime('%Y-%m-%d')
            daily_count = len(self._filter_records_by_date(all_records, today))
            weekly_records = [r for r in all_records if (r.get('created_on') or '')[:10] >= week_start]
            monthly_records = [r for r in all_records if (r.get('created_on') or '')[:10] >= month_start]
            return {
                'daily': {
                    'date': today,
                    'totalDetections': daily_count,
                    'ocrAccuracy': round(ocr_pct, 1),
                    'anomalies': anomalies,
                    'avgProcessingTime': 86,
                },
                'weekly': {
                    'week': f'Last 7 days',
                    'totalDetections': len(weekly_records),
                    'ocrAccuracy': round(ocr_pct, 1),
                    'anomalies': anomalies,
                    'avgProcessingTime': 92,
                },
                'monthly': {
                    'month': datetime.utcnow().strftime('%B %Y'),
                    'totalDetections': len(monthly_records),
                    'ocrAccuracy': round(ocr_pct, 1),
                    'anomalies': anomalies,
                    'avgProcessingTime': 88,
                },
            }
        except Exception as e:
            log.debug("[MetricsServer] _get_reports_from_db failed: %s", e)
            return {
                'daily': {'date': '', 'totalDetections': 0, 'ocrAccuracy': 0, 'anomalies': 0, 'avgProcessingTime': 0},
                'weekly': {'week': '', 'totalDetections': 0, 'ocrAccuracy': 0, 'anomalies': 0, 'avgProcessingTime': 0},
                'monthly': {'month': '', 'totalDetections': 0, 'ocrAccuracy': 0, 'anomalies': 0, 'avgProcessingTime': 0},
            }
    
    def _get_dashboard_data_from_json(self, date_str: str = None) -> Dict:
        """Get dashboard data aggregated from combined_results.json files.
        
        Args:
            date_str: Optional date string in YYYY-MM-DD format to filter by. 
                     If None, defaults to today's date.
        """
        base_dir = Path(__file__).parent.parent / 'out' / 'crops' / 'test-video'
        
        # Parse date filter
        filter_date = None
        if date_str:
            try:
                filter_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                log.info(f"[MetricsServer] Invalid date format: {date_str}, using today")
                filter_date = datetime.now().date()
        else:
            filter_date = datetime.now().date()
        
        # Find all combined_results.json files
        combined_files = []
        if base_dir.exists():
            for folder in base_dir.iterdir():
                if folder.is_dir():
                    json_file = folder / 'combined_results.json'
                    if json_file.exists():
                        combined_files.append(json_file)
        
        if not combined_files:
            # Return empty/default data structure
            return {
                'kpis': {
                    'trailersOnYard': {'value': 0, 'change': '+0', 'icon': '🚛'},
                    'newDetections24h': {'value': 0, 'ocrAccuracy': '0%', 'icon': '📈'},
                    'anomalies': {'value': 0, 'description': 'No anomalies detected', 'icon': '⚠️'},
                    'camerasOnline': {'value': 0, 'degraded': 0, 'icon': '📷'}
                },
                'queueStatus': {'ingestQ': 0, 'ocrQ': 0, 'pubQ': 0},
                'accuracyChart': [],
                'yardUtilization': [],
                'cameraHealth': []
            }
        
        # Read and aggregate all combined results
        all_results = []
        for json_file in combined_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    if isinstance(results, list):
                        all_results.extend(results)
            except Exception as e:
                log.info(f"[MetricsServer] Error reading {json_file}: {e}")
                continue
        
        if not all_results:
            return {
                'kpis': {
                    'trailersOnYard': {'value': 0, 'change': '+0', 'icon': '🚛'},
                    'newDetections24h': {'value': 0, 'ocrAccuracy': '0%', 'icon': '📈'},
                    'anomalies': {'value': 0, 'description': 'No anomalies detected', 'icon': '⚠️'},
                    'camerasOnline': {'value': 0, 'degraded': 0, 'icon': '📷'}
                },
                'queueStatus': {'ingestQ': 0, 'ocrQ': 0, 'pubQ': 0},
                'accuracyChart': [],
                'yardUtilization': [],
                'cameraHealth': []
            }
        
        # Filter results by date if specified
        filtered_results = []
        for result in all_results:
            timestamp = result.get('timestamp', '')
            if timestamp:
                try:
                    result_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                    if result_date == filter_date:
                        filtered_results.append(result)
                except:
                    # If we can't parse timestamp, skip this result when filtering by date
                    if not date_str:
                        filtered_results.append(result)
            else:
                # If no timestamp, only include when not filtering by date
                if not date_str:
                    filtered_results.append(result)
        
        # Use filtered results for calculations
        all_results = filtered_results

        # Calculate KPIs
        unique_tracks = set()
        unique_spots = set()
        ocr_successful = 0
        total_detections = len(all_results)
        low_conf_detections = 0

        for result in all_results:
            track_id = result.get('track_id')
            if track_id:
                unique_tracks.add(track_id)

            spot = result.get('spot', '')
            if spot and spot != 'unknown':
                unique_spots.add(spot)

            ocr_text = result.get('ocr_text', '')
            if ocr_text and ocr_text.strip():
                ocr_successful += 1

            det_conf = result.get('det_conf', 0.0)
            if isinstance(det_conf, (int, float)) and det_conf < 0.5:
                low_conf_detections += 1
        
        # Calculate OCR accuracy
        ocr_accuracy = (ocr_successful / total_detections * 100) if total_detections > 0 else 0
        
        # Get camera health
        cameras = self._get_cameras_with_status()
        online_cameras = sum(1 for c in cameras if c.get('status') == 'online')
        degraded_cameras = sum(1 for c in cameras if c.get('status') == 'degraded')
        
        # Build accuracy chart data (filtered by date)
        accuracy_data = []
        # Group by hour for chart, filter by selected date
        from collections import defaultdict
        hourly_data = defaultdict(lambda: {
            'total': 0, 
            'total_det_conf': 0.0,
            'total_ocr_conf': 0.0,
            'ocr_attempts': 0  # Count of detections where OCR was actually attempted
        })
        
        for result in all_results:
            timestamp = result.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    result_date = dt.date()
                    
                    # Only include results from the selected date (already filtered above, but double-check)
                    if result_date != filter_date:
                        continue
                    
                    hour_key = dt.strftime('%H:00')
                    hourly_data[hour_key]['total'] += 1
                    
                    # For detection accuracy, use the actual confidence value
                    # Detection accuracy = average confidence of detections
                    det_conf = result.get('det_conf', 0.0)
                    if isinstance(det_conf, (int, float)):
                        det_conf_float = float(det_conf)
                        hourly_data[hour_key]['total_det_conf'] += det_conf_float
                    
                    # For OCR accuracy, use the OCR confidence value
                    # Only count detections where OCR was actually attempted
                    ocr_conf = result.get('ocr_conf', 0.0)
                    if isinstance(ocr_conf, (int, float)):
                        ocr_conf_float = float(ocr_conf)
                        # Only count if OCR was attempted (confidence > 0 or text exists)
                        ocr_text = result.get('ocr_text', '') or result.get('text', '')
                        if ocr_conf_float > 0 or (ocr_text and ocr_text.strip()):
                            hourly_data[hour_key]['total_ocr_conf'] += ocr_conf_float
                            hourly_data[hour_key]['ocr_attempts'] += 1
                except Exception as e:
                    # Skip invalid timestamps
                    continue
        
        # Generate data for all 24 hours of today (even if no data)
        # This ensures the chart shows the full day
        for hour in range(24):
            hour_key = f"{hour:02d}:00"
            data = hourly_data.get(hour_key, {
                'total': 0, 
                'total_det_conf': 0.0,
                'total_ocr_conf': 0.0,
                'ocr_attempts': 0
            })
            
            # Calculate detection accuracy (average detection confidence * 100)
            if data['total'] > 0:
                avg_det_conf = data['total_det_conf'] / data['total']
                detection_accuracy = avg_det_conf * 100
            else:
                detection_accuracy = 0
            
            # Calculate OCR accuracy (average OCR confidence * 100)
            # Only average over detections where OCR was actually attempted
            # This gives a more accurate representation of OCR quality
            if data['ocr_attempts'] > 0:
                avg_ocr_conf = data['total_ocr_conf'] / data['ocr_attempts']
                ocr_accuracy = avg_ocr_conf * 100
            elif data['total'] > 0:
                # If there are detections but no OCR attempts, accuracy is 0%
                ocr_accuracy = 0
            else:
                # No data for this hour
                ocr_accuracy = 0
            
            accuracy_data.append({
                'time': hour_key,
                'detection': round(detection_accuracy, 1),
                'ocr': round(ocr_accuracy, 1)
            })
        
        # Build yard utilization data
        spot_counts = defaultdict(int)
        for result in all_results:
            spot = result.get('spot', '')
            if spot and spot != 'unknown':
                spot_counts[spot] += 1
        
        utilization_data = []
        for spot, count in sorted(spot_counts.items()):
            utilization_data.append({
                'lane': spot,
                'utilization': count
            })
        
        return {
            'kpis': {
                'trailersOnYard': {
                    'value': len(unique_tracks),
                    'change': f'+{len(unique_tracks)}',
                    'icon': '🚛'
                },
                'newDetections24h': {
                    'value': total_detections,
                    'ocrAccuracy': f'{ocr_accuracy:.1f}%',
                    'icon': '📈'
                },
                'anomalies': {
                    'value': low_conf_detections,
                    'description': f'{low_conf_detections} low confidence detections',
                    'icon': '⚠️'
                },
                'camerasOnline': {
                    'value': online_cameras,
                    'degraded': degraded_cameras,
                    'icon': '📷'
                }
            },
            'queueStatus': {
                'ingestQ': 0,
                'ocrQ': 0,
                'pubQ': 0
            },
            'accuracyChart': accuracy_data,  # Direct array, not wrapped in 'data'
            'yardUtilization': utilization_data,  # Direct array, not wrapped in 'data'
            'cameraHealth': cameras
        }
    
    def _get_events_from_json(self, limit: int = 1000, date_str: str = None) -> List[Dict]:
        """Get events from combined_results.json files.
        
        Args:
            limit: Maximum number of events to return
            date_str: Optional date string in YYYY-MM-DD format to filter by.
                     If None, returns all events (up to limit).
        """
        base_dir = Path(__file__).parent.parent / 'out' / 'crops' / 'test-video'
        
        # Parse date filter
        filter_date = None
        if date_str:
            try:
                filter_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                log.info(f"[MetricsServer] Invalid date format: {date_str}, returning all events")
                filter_date = None
        
        # Find all combined_results.json files
        combined_files = []
        if base_dir.exists():
            for folder in base_dir.iterdir():
                if folder.is_dir():
                    json_file = folder / 'combined_results.json'
                    if json_file.exists():
                        combined_files.append(json_file)
        
        all_events = []
        for json_file in combined_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    if isinstance(results, list):
                        for result in results:
                            # Filter by date if specified
                            if filter_date:
                                timestamp = result.get('timestamp', '')
                                if timestamp:
                                    try:
                                        result_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                                        if result_date != filter_date:
                                            continue
                                    except:
                                        continue
                                else:
                                    continue  # Skip events without timestamp when filtering by date
                            
                            # Transform to event format expected by frontend
                            # Frontend expects 'text' for OCR text, not 'ocr_text'
                            ocr_text = result.get('ocr_text', '') or result.get('text', '')
                            
                            # Extract GPS coordinates - check direct fields first, then world_coords
                            lat = result.get('lat')
                            lon = result.get('lon')
                            world_coords = result.get('world_coords')
                            
                            # Extract from world_coords if lat/lon not directly available
                            # world_coords format: [lat, lon] when GPS coordinates are available
                            if (lat is None or lon is None) and world_coords:
                                if isinstance(world_coords, (list, tuple)) and len(world_coords) >= 2:
                                    try:
                                        coord1, coord2 = float(world_coords[0]), float(world_coords[1])
                                        # Check if coordinates look like GPS (lat/lon) vs meters
                                        # GPS: lat is -90 to 90, lon is -180 to 180
                                        # Meters: typically much larger values (hundreds or thousands)
                                        if -90.0 <= coord1 <= 90.0 and -180.0 <= coord2 <= 180.0:
                                            lat = coord1
                                            lon = coord2
                                        # Also check if world_coords might be stored as [lon, lat] (less common)
                                        elif -90.0 <= coord2 <= 90.0 and -180.0 <= coord1 <= 180.0:
                                            lat = coord2
                                            lon = coord1
                                    except (ValueError, TypeError) as e:
                                        # If conversion fails, leave lat/lon as None
                                        pass
                            
                            event = {
                                'ts_iso': result.get('timestamp', ''),
                                'camera_id': result.get('camera_id', 'N/A'),
                                'track_id': result.get('track_id', 'N/A'),
                                'text': ocr_text,  # Frontend expects 'text' for OCR result
                                'conf': result.get('det_conf', 0.0),
                                'spot': result.get('spot', 'unknown'),
                                'ocr_conf': result.get('ocr_conf', 0.0),
                                'frame_count': result.get('frame_count', 0),
                                'bbox': result.get('bbox', []),
                                'world_coords': world_coords,
                                'lat': lat,
                                'lon': lon,
                                # Keep original fields for reference
                                'ocr_text': ocr_text,
                                'det_conf': result.get('det_conf', 0.0)
                            }
                            all_events.append(event)
            except Exception as e:
                log.info(f"[MetricsServer] Error reading {json_file}: {e}")
                continue
        
        # Sort by timestamp (most recent first)
        all_events.sort(key=lambda x: x.get('ts_iso', ''), reverse=True)
        
        # Return limited results
        return all_events[:limit]
    
    def _get_inventory_from_json(self) -> Dict:
        """Get inventory data from combined_results.json files."""
        base_dir = Path(__file__).parent.parent / 'out' / 'crops' / 'test-video'
        
        # Find all combined_results.json files
        combined_files = []
        if base_dir.exists():
            for folder in base_dir.iterdir():
                if folder.is_dir():
                    json_file = folder / 'combined_results.json'
                    if json_file.exists():
                        combined_files.append(json_file)
        
        if not combined_files:
            return {
                'trailers': [],
                'stats': {
                    'total': 0,
                    'parked': 0,
                    'inTransit': 0,
                    'anomalies': 0
                }
            }
        
        # Read and aggregate all combined results
        all_results = []
        for json_file in combined_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    if isinstance(results, list):
                        all_results.extend(results)
            except Exception as e:
                log.info(f"[MetricsServer] Error reading {json_file}: {e}")
                continue
        
        # Group by track_id to get unique trailers
        from collections import defaultdict
        trailers_by_track = defaultdict(list)
        
        for result in all_results:
            track_id = result.get('track_id')
            if track_id:
                trailers_by_track[track_id].append(result)
        
        # Build trailer list (one per unique track_id)
        trailers = []
        for track_id, results_list in trailers_by_track.items():
            # Get the most recent result for this track
            most_recent = max(results_list, key=lambda x: x.get('timestamp', ''))
            
            ocr_text = most_recent.get('ocr_text', '') or most_recent.get('text', '')
            spot = most_recent.get('spot', 'unknown')
            timestamp = most_recent.get('timestamp', '')
            ocr_conf = most_recent.get('ocr_conf', 0.0)
            det_conf = most_recent.get('det_conf', 0.0)
            
            # Extract GPS coordinates - check direct fields first, then world_coords
            lat = most_recent.get('lat')
            lon = most_recent.get('lon')
            world_coords = most_recent.get('world_coords')
            
            # Extract from world_coords if lat/lon not directly available
            # world_coords format: [lat, lon] when GPS coordinates are available
            if (lat is None or lon is None) and world_coords:
                if isinstance(world_coords, (list, tuple)) and len(world_coords) >= 2:
                    try:
                        coord1, coord2 = float(world_coords[0]), float(world_coords[1])
                        # Check if coordinates look like GPS (lat/lon) vs meters
                        # GPS: lat is -90 to 90, lon is -180 to 180
                        # Meters: typically much larger values (hundreds or thousands)
                        if -90.0 <= coord1 <= 90.0 and -180.0 <= coord2 <= 180.0:
                            lat = coord1
                            lon = coord2
                        # Also check if world_coords might be stored as [lon, lat] (less common)
                        elif -90.0 <= coord2 <= 90.0 and -180.0 <= coord1 <= 180.0:
                            lat = coord2
                            lon = coord1
                    except (ValueError, TypeError) as e:
                        # If conversion fails, leave lat/lon as None
                        pass
            
            # Determine status based on spot and confidence
            if spot == 'unknown' or not spot:
                status = 'In Transit'
            else:
                status = 'Parked'
            
            # Use track_id as trailer ID, or generate one
            trailer_id = f"T{track_id}" if track_id else f"T{len(trailers) + 1}"
            
            trailer = {
                'id': trailer_id,
                'plate': ocr_text if ocr_text.strip() else 'N/A',
                'spot': spot if spot != 'unknown' else 'N/A',
                'status': status,
                'detectedAt': timestamp,
                'ocrConfidence': float(ocr_conf) if ocr_conf else 0.0,
                'lat': lat,
                'lon': lon
            }
            
            trailers.append(trailer)
        
        # Calculate stats
        total = len(trailers)
        parked = sum(1 for t in trailers if t['status'] == 'Parked')
        in_transit = sum(1 for t in trailers if t['status'] == 'In Transit')
        anomalies = sum(1 for r in all_results if r.get('det_conf', 1.0) < 0.5)
        
        return {
            'trailers': trailers,
            'stats': {
                'total': total,
                'parked': parked,
                'inTransit': in_transit,
                'anomalies': anomalies
            }
        }
    
    def _get_cameras_with_status(self, force_check: bool = False) -> List[Dict]:
        """
        Get cameras from cameras.yaml and check their status.
        Uses caching to avoid testing cameras on every request.
        
        Args:
            force_check: If True, force a fresh camera status check (bypass cache)
        """
        import yaml
        import time
        
        # Cache camera status for 30 seconds to avoid repeated tests
        if not hasattr(self, '_camera_cache'):
            self._camera_cache = {'data': [], 'timestamp': 0}
        
        cache_ttl = 30  # Cache for 30 seconds
        current_time = time.time()
        
        # Return cached data if still valid and not forcing a check
        if not force_check and (current_time - self._camera_cache['timestamp']) < cache_ttl:
            return self._camera_cache['data']
        
        config_path = Path(__file__).parent.parent / 'config' / 'cameras.yaml'
        if not config_path.exists():
            return []
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            log.info(f"[MetricsServer] Error reading cameras.yaml: {e}")
            return []
        
        cameras = config.get('cameras', [])
        globals_cfg = config.get('globals', {})
        use_gstreamer = globals_cfg.get('use_gstreamer', False)
        
        camera_list = []
        for camera in cameras:
            camera_id = camera.get('id', 'unknown')
            rtsp_url = camera.get('rtsp_url', '')
            width = camera.get('width', 1920)
            height = camera.get('height', 1080)
            fps_cap = camera.get('fps_cap', 30)
            
            # Determine camera type
            if rtsp_url.isdigit() or rtsp_url == '0':
                camera_type = 'USB'
                device_index = int(rtsp_url)
            elif rtsp_url.startswith('rtsp://'):
                camera_type = 'RTSP'
            else:
                camera_type = 'Unknown'
            
            # Test camera connectivity (only if force_check or cache expired)
            status = 'offline'
            fps = 0
            latency = 0
            
            try:
                # Test camera connectivity (warnings are suppressed in test_stream)
                from app.rtsp import test_stream
                is_accessible = test_stream(rtsp_url, width, height, use_gstreamer)
                
                if is_accessible:
                    status = 'online'
                    fps = fps_cap  # Use configured FPS as default
                    latency = 50  # Default latency estimate
                else:
                    status = 'offline'
                    fps = 0
                    latency = 0
            except Exception:
                # Camera is not accessible - mark as offline
                # This is expected when cameras are not connected
                status = 'offline'
                fps = 0
                latency = 0
            
            camera_info = {
                'id': camera_id,
                'name': camera_id,
                'type': camera_type,
                'rtsp_url': rtsp_url,
                'status': status,
                'fps': fps,
                'latency': latency,
                'width': width,
                'height': height,
                'fps_cap': fps_cap
            }
            
            camera_list.append(camera_info)
        
        # Update cache
        self._camera_cache = {'data': camera_list, 'timestamp': current_time}
        
        return camera_list
    
    def _get_metrics(self) -> Dict:
        """Get current metrics snapshot."""
        return {
            'cameras': metrics_registry['cameras'].copy(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _get_recent_events(self, camera_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get recent events from CSV file."""
        if self.csv_logger is None:
            return []
        
        events = []
        try:
            csv_path = Path(self.csv_logger.get_today_filename())
            if not csv_path.exists():
                return []
            
            import csv
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if camera_id and row.get('camera_id') != camera_id:
                        continue
                    
                    # Normalize row data: ensure all values are JSON-serializable
                    # Convert all values to JSON-safe types (no None to avoid comparison errors)
                    normalized_row = {}
                    for key, value in row.items():
                        # Ensure value is not None and convert to appropriate type
                        # CSV DictReader should return strings, but handle all edge cases
                        if value is None:
                            normalized_row[key] = ''
                        elif value == '':
                            normalized_row[key] = ''
                        else:
                            # Non-empty value - convert to appropriate type
                            try:
                                value_str = str(value).strip()
                                if not value_str:
                                    normalized_row[key] = ''
                                elif key in ['x_world', 'y_world', 'conf']:
                                    # Convert numeric fields to float, use 0.0 for empty/invalid
                                    try:
                                        float_val = float(value_str)
                                        # Check for NaN or infinity
                                        if float_val != float_val or not (-1e308 < float_val < 1e308):
                                            normalized_row[key] = 0.0
                                        else:
                                            normalized_row[key] = float_val
                                    except (ValueError, TypeError, OverflowError):
                                        normalized_row[key] = 0.0
                                elif key == 'track_id':
                                    # Convert track_id to int, use 0 for empty/invalid
                                    try:
                                        normalized_row[key] = int(value_str)
                                    except (ValueError, TypeError):
                                        normalized_row[key] = 0
                                else:
                                    # Keep as string for text fields
                                    normalized_row[key] = value_str
                            except Exception:
                                # Fallback: use empty string for any conversion error
                                normalized_row[key] = ''
                    
                    events.append(normalized_row)
            
            # Final safety check: ensure all events are JSON-serializable
            # Convert any remaining None values to safe defaults
            for event in events:
                for key, value in list(event.items()):
                    if value is None:
                        event[key] = '' if key not in ['x_world', 'y_world', 'conf'] else 0.0
            
            # Return last N events
            return events[-limit:]
        except Exception as e:
            log.info(f"Error reading events: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def update_camera_metrics(self, camera_id: str, fps_ema: float, frames_processed_count: int, 
                             last_publish: Optional[datetime] = None, queue_depth: int = 0):
        """
        Update metrics for a camera.
        
        Args:
            camera_id: Camera identifier
            fps_ema: EMA of frames per second
            frames_processed_count: Total frames processed
            last_publish: Last publish timestamp
            queue_depth: Current queue depth
        """
        if camera_id not in metrics_registry['cameras']:
            metrics_registry['cameras'][camera_id] = {}
        
        metrics_registry['cameras'][camera_id].update({
            'fps_ema': fps_ema,
            'frames_processed': frames_processed_count,
            'last_publish': last_publish.isoformat() if last_publish else None,
            'queue_depth': queue_depth
        })
        
        # Update Prometheus metrics
        fps_gauge.labels(camera_id=camera_id).set(fps_ema)
        frames_processed.labels(camera_id=camera_id)._value._value = frames_processed_count
        if last_publish:
            last_publish_time.labels(camera_id=camera_id).set(last_publish.timestamp())
        queue_depth_gauge.set(queue_depth)
    
    def start(self):
        """Start metrics server in background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        log.info(f"Metrics server started on port {self.port}")
    
    def _run_server(self):
        """Run Flask server."""
        self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False, threaded=True)
    
    def _generate_mjpeg(self, camera_id: str):
        """
        Generate MJPEG stream for a camera.
        
        Args:
            camera_id: Camera identifier
            
        Yields:
            JPEG frame data
        """
        no_frame_count = 0
        while True:
            if self.frame_storage is None:
                time.sleep(0.1)
                continue

            # Stall while a VLM is being loaded into unified DRAM. On Jetson Orin,
            # 20+ concurrent JPEG-encode buffers (one per /stream/<camera> request
            # the dashboard makes per second) push peak memory over the limit
            # right at the end of Qwen3-VL weight loading and the kernel OOM-kills
            # the process. While loading, we just keep the connection open and
            # send no frames; the browser will pick up where it left off once the
            # flag clears.
            if getattr(self.frame_storage, "gpu_load_in_progress", False):
                time.sleep(0.5)
                continue

            # Same rationale: while an offline gate-test (or yard-test) job is
            # running, the processing_queue holds ``gpu_lock`` for the whole
            # video phase AND the whole OCR phase. Each /stream/<camera> poll
            # otherwise allocates a fresh BGR copy + JPEG-encode buffer per
            # iteration (~2.6 MB at 720p) and that transient pressure has been
            # enough to OOM the Qwen-VL OCR mid-batch. Pause the stream for
            # the duration of the offline test; the dashboard preview freezes
            # on the last sent frame and resumes when the lock is released.
            try:
                pq_ref = getattr(self.frame_storage, "processing_queue", None)
                gpu_lock_ref = getattr(pq_ref, "gpu_lock", None) if pq_ref is not None else None
                if gpu_lock_ref is not None and gpu_lock_ref.locked():
                    time.sleep(0.5)
                    continue
            except Exception:
                # Never let a bookkeeping error break the live stream.
                pass

            # Get latest frame (thread-safe)
            frame = None
            try:
                with self.frame_storage.frame_lock:
                    if camera_id in self.frame_storage.latest_frames:
                        frame = self.frame_storage.latest_frames[camera_id].copy()
            except Exception as e:
                log.info(f"[MetricsServer] Error accessing frame storage for {camera_id}: {e}")
                time.sleep(0.1)
                continue

            if frame is not None:
                try:
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                        no_frame_count = 0
                    else:
                        no_frame_count += 1
                except Exception as e:
                    log.info(f"[MetricsServer] Error encoding frame for {camera_id}: {e}")
                    no_frame_count += 1
            else:
                no_frame_count += 1
                # If no frame for a while, wait longer
                if no_frame_count > 10:
                    time.sleep(0.5)
                    continue
            
            # Limit frame rate to ~30 FPS
            time.sleep(1.0 / 30.0)
    
    def _generate_processed_mjpeg(self):
        """Generate MJPEG stream from processed video frames."""
        current_frame_num = 0
        last_successful_frame = -1
        no_frame_count = 0
        consecutive_failures = 0
        
        while True:
            if self.video_processor is None:
                time.sleep(0.1)
                continue
            
            # Get latest frame number and processing status
            results = self.video_processor.get_results()
            frames_processed = results.get('frames_processed', 0)
            is_processing = self.video_processor.is_processing()
            
            if frames_processed > 0:
                # Try to get the current frame we want to display
                # Start from frame 0 and progress through frames sequentially
                frame = None
                
                # Try current frame first
                if current_frame_num < frames_processed:
                    frame = self.video_processor.get_frame(current_frame_num)
                
                # If current frame not available, try a few frames ahead
                if frame is None and current_frame_num < frames_processed:
                    for offset in range(1, min(10, frames_processed - current_frame_num)):
                        try_frame = self.video_processor.get_frame(current_frame_num + offset)
                        if try_frame is not None:
                            frame = try_frame
                            current_frame_num = current_frame_num + offset
                            break
                
                # If still no frame, try previous frames
                if frame is None and current_frame_num > 0:
                    for offset in range(1, min(10, current_frame_num + 1)):
                        try_frame = self.video_processor.get_frame(current_frame_num - offset)
                        if try_frame is not None:
                            frame = try_frame
                            current_frame_num = current_frame_num - offset
                            break
                
                if frame is not None:
                    try:
                        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                            last_successful_frame = current_frame_num
                            no_frame_count = 0
                            consecutive_failures = 0
                            
                            # Advance to next frame for next iteration
                            current_frame_num += 1
                            
                            # If we've reached the end and processing is done, loop back to start
                            if not is_processing and current_frame_num >= frames_processed:
                                current_frame_num = 0  # Loop back to start
                        else:
                            consecutive_failures += 1
                    except Exception as e:
                        log.info(f"[MetricsServer] Error encoding frame {current_frame_num}: {e}")
                        consecutive_failures += 1
                else:
                    no_frame_count += 1
                    consecutive_failures += 1
                    # If we can't find a frame, try to advance anyway
                    if current_frame_num < frames_processed:
                        current_frame_num += 1
            else:
                no_frame_count += 1
                consecutive_failures += 1
            
            # If processing is done and we've shown all frames, loop back
            if not is_processing and frames_processed > 0 and current_frame_num >= frames_processed:
                current_frame_num = 0  # Loop back to start for continuous playback
            
            # Limit frame rate - faster when processing, slower when done
            if is_processing:
                time.sleep(1.0 / 15.0)  # 15 FPS when processing
            else:
                # When done, play at normal speed (loop)
                time.sleep(1.0 / 10.0)  # 10 FPS for playback
    
    def stop(self):
        """Stop metrics server."""
        self.running = False
        log.info("Metrics server stopped")

