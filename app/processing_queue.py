"""
Processing Queue Manager

Manages sequential processing of video chunks and OCR to prevent GPU memory conflicts.
Ensures only one GPU operation runs at a time (video processing OR OCR, never both).

Priority System:
- OCR has PRIORITY: When OCR is processing, video processing waits in queue
- Video processing waits: Checks OCR status before acquiring GPU lock
- Sequential execution: Only one GPU operation at a time

Workflow (default - per-chunk OCR):
1. Video chunk saved → Queue video processing job
2. Video processing waits if OCR is active
3. Video processing completes → Queue OCR job on saved crops
4. OCR processing (has priority, video waits if queued)
5. OCR completes → Ready for server upload

Workflow (defer_ocr=True - OCR at end only):
1. Video chunk saved → Queue video processing job only (no OCR loaded)
2. Video processing runs (YOLO + tracking, save crops)
3. When recording stopped AND video queue empty → on_video_queue_drained(pending_jobs)
4. App loads OCR and runs batch OCR on all pending crops once
"""

import threading
import queue
import time
from pathlib import Path
from typing import Optional, Dict, Callable, Any, List
from datetime import datetime
import gc

from app.app_logger import get_logger

log = get_logger(__name__)

# Try to import torch for GPU memory management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ProcessingQueueManager:
    """
    Manages sequential processing queue for video and OCR operations.
    Ensures GPU is only used by one operation at a time.
    
    Priority: OCR has priority over video processing.
    - When OCR is processing, video processing jobs wait in queue
    - Video processing only starts when OCR is not active
    - GPU lock ensures only one operation uses GPU at a time
    """
    
    def __init__(self, video_processor, ocr, preprocessor=None, on_video_complete: Optional[Callable] = None,
                 on_ocr_complete: Optional[Callable] = None, defer_ocr: bool = False,
                 on_video_queue_drained: Optional[Callable[[List[Dict]], None]] = None):
        """
        Initialize processing queue manager.

        Args:
            video_processor: VideoProcessor instance
            ocr: OCR recognizer instance (for batch processing). Can be None if defer_ocr=True.
            preprocessor: Optional ImagePreprocessor instance
            on_video_complete: Optional callback(video_path, crops_dir, results) when video processing completes
            on_ocr_complete: Optional callback(video_path, crops_dir, ocr_results) when OCR completes
            defer_ocr: If True, do not load/run OCR per chunk; accumulate jobs and call on_video_queue_drained
                       when recording has stopped and video queue is empty (OCR runs once at end).
            on_video_queue_drained: When defer_ocr=True, called with list of {video_path, crops_dir, camera_id}
                                   when recording stopped and video queue has drained. App should run OCR then.
        """
        self.video_processor = video_processor
        self.ocr = ocr
        self.preprocessor = preprocessor
        self.on_video_complete = on_video_complete
        self.on_ocr_complete = on_ocr_complete
        self.defer_ocr = defer_ocr
        self.on_video_queue_drained = on_video_queue_drained

        # Deferred OCR: accumulate (video_path, crops_dir, camera_id) until recording stopped and queue empty
        self.recording_stopped = False
        self.pending_ocr_jobs: List[Dict[str, Any]] = []
        self._pending_lock = threading.Lock()

        # Processing queues
        self.video_queue = queue.Queue()
        self.ocr_queue = queue.Queue()

        # GPU lock - ensures only one GPU operation at a time
        self.gpu_lock = threading.Lock()

        # Processing state (thread-safe flags)
        self.processing_video = False
        self.processing_ocr = False
        self.running = True
        self.ocr_stop_requested = False

        # Lock for state flags (for thread-safe checking)
        self.state_lock = threading.Lock()

        # Statistics
        self.stats = {
            'videos_queued': 0,
            'videos_processed': 0,
            'ocr_jobs_queued': 0,
            'ocr_jobs_processed': 0,
            'errors': 0
        }

        # Start video worker always; start OCR worker only when not deferring OCR
        self.video_worker_thread = threading.Thread(target=self._video_worker, daemon=True)
        self.video_worker_thread.start()
        if not defer_ocr:
            self.ocr_worker_thread = threading.Thread(target=self._ocr_worker, daemon=True)
            self.ocr_worker_thread.start()
            log.info("Initialized with GPU lock protection (per-chunk OCR)")
        else:
            self.ocr_worker_thread = None
            log.info("Initialized with GPU lock protection (deferred OCR: run once after all video processing)")
    
    def queue_video_processing(
        self,
        video_path: str,
        camera_id: str,
        gps_log_path: Optional[str] = None,
        detect_every_n: int = 5,
        detection_mode: str = 'trailer',
        on_ocr_complete: Optional[Callable[..., None]] = None,
    ):
        """
        Queue a video for processing.
        
        Args:
            video_path: Path to video file
            camera_id: Camera identifier
            gps_log_path: Optional path to GPS log file
            detect_every_n: Run detector every N frames
            detection_mode: 'trailer' or 'car'
            on_ocr_complete: Optional per-job OCR callback (video_path, crops_dir, combined_results);
                overrides manager default for this video's OCR job only.
        """
        job = {
            'video_path': video_path,
            'camera_id': camera_id,
            'gps_log_path': gps_log_path,
            'detect_every_n': detect_every_n,
            'detection_mode': detection_mode,
            'on_ocr_complete': on_ocr_complete,
            'queued_at': datetime.now().isoformat()
        }
        
        self.video_queue.put(job)
        self.stats['videos_queued'] += 1
        log.info(f"[ProcessingQueueManager] Queued video processing: {Path(video_path).name} (queue size: {self.video_queue.qsize()})")
    
    def _video_worker(self):
        """
        Worker thread that processes videos sequentially.
        Waits for OCR to complete before processing videos (OCR has priority).
        """
        from app.batch_ocr_processor import BatchOCRProcessor
        
        while self.running:
            try:
                # Get next video job (blocking with timeout to allow checking self.running)
                try:
                    job = self.video_queue.get(timeout=1.0)
                except queue.Empty:
                    # Idle path: also check the deferred-OCR drain condition here
                    # so the gate-test flow still triggers OCR when
                    # ``notify_recording_stopped()`` is called *after* the last
                    # video has already finished. Without this check the drain
                    # callback only fires at the end of each video iteration,
                    # so a late ``recording_stopped=True`` would never trigger.
                    if self.defer_ocr and self.recording_stopped:
                        with self._pending_lock:
                            jobs = list(self.pending_ocr_jobs)
                            self.pending_ocr_jobs.clear()
                        if jobs and self.on_video_queue_drained:
                            log.info(
                                "[ProcessingQueueManager] Idle drain: video queue empty + recording_stopped=True; "
                                "freeing GPU before deferred OCR on %s job(s)",
                                len(jobs),
                            )
                            self._cleanup_gpu_memory()
                            try:
                                self.on_video_queue_drained(jobs)
                            except Exception as e:
                                log.exception(
                                    f"[ProcessingQueueManager] Error in on_video_queue_drained (idle path): {e}"
                                )
                    continue
                
                video_path = job['video_path']
                camera_id = job['camera_id']
                gps_log_path = job.get('gps_log_path')
                detect_every_n = job.get('detect_every_n', 5)
                detection_mode = job.get('detection_mode', 'trailer')
                job_on_ocr_complete = job.get('on_ocr_complete')
                
                # CRITICAL: Wait for OCR to complete before processing video
                # OCR has priority - video processing must wait until OCR is completely done
                wait_start = time.time()
                with self.state_lock:
                    ocr_active = self.processing_ocr
                
                # Wait loop: Keep waiting while OCR is active
                while ocr_active and self.running:
                    elapsed = time.time() - wait_start
                    if elapsed > 1.0:  # Log every second while waiting
                        log.info(f"[ProcessingQueueManager] Video processing queued, waiting for OCR to complete: {Path(video_path).name} (waited {elapsed:.1f}s)")
                        wait_start = time.time()  # Reset to avoid spam
                    time.sleep(0.1)  # Check every 100ms
                    
                    # Re-check OCR status
                    with self.state_lock:
                        ocr_active = self.processing_ocr
                
                if not self.running:
                    # Put job back in queue if shutting down
                    self.video_queue.put(job)
                    break
                
                # At this point, OCR should be done. Proceed with video processing.
                log.info(f"[ProcessingQueueManager] OCR complete, starting video processing: {Path(video_path).name}")
                
                # Acquire GPU lock for video processing
                # This ensures no OCR can start while we're processing video
                with self.gpu_lock:
                    # Double-check OCR didn't start (shouldn't happen, but safety check)
                    with self.state_lock:
                        if self.processing_ocr:
                            log.error("OCR became active while acquiring GPU lock - this should not happen")
                            # Put job back and wait again
                            self.video_queue.put(job)
                            continue
                        
                        # All clear - start video processing
                        self.processing_video = True
                    try:
                        # Update video processor GPS log path and load GPS log if provided
                        if gps_log_path and hasattr(self.video_processor, 'gps_log_path'):
                            self.video_processor.gps_log_path = gps_log_path
                            try:
                                from app.container_utils import load_gps_log
                                if Path(gps_log_path).exists():
                                    self.video_processor.gps_log = load_gps_log(gps_log_path)
                                    log.info("[ProcessingQueueManager] Loaded GPS log for %s: %s entries", Path(video_path).name, len(self.video_processor.gps_log))
                                else:
                                    self.video_processor.gps_log = {}
                                    log.info("[ProcessingQueueManager] GPS log file not found: %s", gps_log_path)
                            except Exception as e:
                                log.warning("[ProcessingQueueManager] Failed to load GPS log %s: %s", gps_log_path, e)
                                self.video_processor.gps_log = {}
                        
                        # Set detection mode (car vs trailer) so the correct detector is used
                        if hasattr(self.video_processor, 'set_detection_mode'):
                            self.video_processor.set_detection_mode(detection_mode)
                        
                        # Process video (detection, tracking, save crops - NO OCR)
                        results = None
                        crops_dir = None
                        
                        # Process video and collect results
                        for frame_num, processed_frame, events in self.video_processor.process_video(
                            video_path, 
                            camera_id=camera_id,
                            detect_every_n=detect_every_n
                        ):
                            # Just consume the generator - processing happens here
                            pass
                        
                        # Get results after processing
                        if hasattr(self.video_processor, 'get_results'):
                            results = self.video_processor.get_results()
                        
                        # Get crops directory
                        if hasattr(self.video_processor, 'crops_dir') and self.video_processor.crops_dir:
                            crops_dir = str(self.video_processor.crops_dir)
                        
                        # Cleanup GPU memory after video processing
                        self._cleanup_gpu_memory()
                        
                        log.info(f"[ProcessingQueueManager] Video processing complete: {Path(video_path).name}")
                        log.info(f"  - Crops saved to: {crops_dir}")
                        
                        self.stats['videos_processed'] += 1
                        
                        # Call completion callback
                        if self.on_video_complete:
                            try:
                                self.on_video_complete(video_path, crops_dir, results)
                            except Exception as e:
                                log.info(f"[ProcessingQueueManager] Error in video completion callback: {e}")
                        
                        # Queue OCR job if crops were saved (or accumulate for deferred OCR)
                        if crops_dir and Path(crops_dir).exists():
                            if self.defer_ocr:
                                with self._pending_lock:
                                    self.pending_ocr_jobs.append({
                                        'video_path': video_path,
                                        'crops_dir': crops_dir,
                                        'camera_id': camera_id
                                    })
                                log.info(f"[ProcessingQueueManager] Deferred OCR: added to pending ({len(self.pending_ocr_jobs)} total)")
                            else:
                                self._queue_ocr_job(
                                    video_path,
                                    crops_dir,
                                    camera_id,
                                    on_ocr_complete=job_on_ocr_complete,
                                )
                        else:
                            log.info(f"[ProcessingQueueManager] No crops directory found, skipping OCR")
                    
                    except Exception as e:
                        log.info(f"[ProcessingQueueManager] Error processing video {video_path}: {e}")
                        import traceback
                        traceback.print_exc()
                        self.stats['errors'] += 1
                        self._cleanup_gpu_memory()
                    
                    finally:
                        with self.state_lock:
                            self.processing_video = False
                
                self.video_queue.task_done()
                
                # When defer_ocr: if recording stopped and queue is empty, free GPU then trigger deferred OCR
                if self.defer_ocr and self.video_queue.empty() and self.recording_stopped:
                    with self._pending_lock:
                        jobs = list(self.pending_ocr_jobs)
                        self.pending_ocr_jobs.clear()
                    if jobs and self.on_video_queue_drained:
                        log.info(f"[ProcessingQueueManager] Video queue drained; freeing GPU before deferred OCR on {len(jobs)} job(s)")
                        self._cleanup_gpu_memory()
                        try:
                            self.on_video_queue_drained(jobs)
                        except Exception as e:
                            log.exception(f"[ProcessingQueueManager] Error in on_video_queue_drained: {e}")
                
            except Exception as e:
                log.info(f"[ProcessingQueueManager] Error in video worker thread: {e}")
                import traceback
                traceback.print_exc()
                self.stats['errors'] += 1
                time.sleep(1)  # Prevent tight error loop
    
    def _queue_ocr_job(
        self,
        video_path: str,
        crops_dir: str,
        camera_id: str,
        on_ocr_complete: Optional[Callable] = None,
    ):
        """Queue OCR processing for a video's crops."""
        job = {
            'video_path': video_path,
            'crops_dir': crops_dir,
            'camera_id': camera_id,
            'on_ocr_complete': on_ocr_complete,
            'queued_at': datetime.now().isoformat()
        }
        
        self.ocr_queue.put(job)
        self.stats['ocr_jobs_queued'] += 1
        log.info(f"[ProcessingQueueManager] Queued OCR processing: {Path(crops_dir).name} (queue size: {self.ocr_queue.qsize()})")
    
    def _should_stop_ocr(self) -> bool:
        """Thread-safe check for OCR stop request (e.g. user clicked Stop)."""
        with self.state_lock:
            return self.ocr_stop_requested
    
    def request_ocr_stop(self):
        """Request that current OCR job stop and no further jobs run (call before clearing queue)."""
        with self.state_lock:
            self.ocr_stop_requested = True
    
    def clear_video_queue(self):
        """Remove all pending video jobs so stop takes effect immediately."""
        cleared = 0
        while True:
            try:
                self.video_queue.get_nowait()
                self.video_queue.task_done()
                cleared += 1
            except queue.Empty:
                break
        if cleared:
            log.info(f"[ProcessingQueueManager] Cleared {cleared} video job(s) from queue")
    
    def clear_ocr_queue(self):
        """Remove all pending OCR jobs."""
        cleared = 0
        while not self.ocr_queue.empty():
            try:
                self.ocr_queue.get_nowait()
                self.ocr_queue.task_done()
                cleared += 1
            except queue.Empty:
                break
        if cleared:
            log.info(f"[ProcessingQueueManager] Cleared {cleared} OCR job(s) from queue")
        return cleared

    def set_ocr(self, ocr):
        """Set OCR instance (used when OCR was deferred and is initialized after video processing)."""
        self.ocr = ocr
        log.info("[ProcessingQueueManager] OCR instance set (deferred OCR)")

    def queue_ocr_jobs(self, jobs: List[Dict[str, Any]]):
        """Queue multiple OCR jobs (e.g. from on_video_queue_drained callback).

        Each job dict may contain: video_path, crops_dir, camera_id, and an
        optional ``on_ocr_complete`` callback that will fire after THIS job's
        OCR finishes (so callers like the gate-test endpoint can attach a
        DB-store callback per video).
        """
        for j in jobs:
            self._queue_ocr_job(
                j.get('video_path', ''),
                j.get('crops_dir', ''),
                j.get('camera_id', 'test-video'),
                on_ocr_complete=j.get('on_ocr_complete'),
            )

    def start_ocr_worker_if_deferred(self):
        """Start the OCR worker thread when it was not started at init (defer_ocr=True). Call after set_ocr and queue_ocr_jobs."""
        if self.ocr_worker_thread is not None:
            return
        self.ocr_worker_thread = threading.Thread(target=self._ocr_worker, daemon=True)
        self.ocr_worker_thread.start()
        log.info("[ProcessingQueueManager] Started OCR worker (deferred OCR)")

    def notify_recording_stopped(self):
        """Call when recording has been stopped (no more chunks will be queued). Used with defer_ocr."""
        self.recording_stopped = True
        log.info("[ProcessingQueueManager] Recording stopped; deferred OCR will run when video queue drains")

    def notify_recording_started(self):
        """Call when recording starts again (reset so deferred OCR runs for this session)."""
        self.recording_stopped = False
    
    def _ocr_worker(self):
        """
        Worker thread that processes OCR jobs sequentially.
        OCR has priority - video processing will wait for OCR to complete.
        """
        from app.batch_ocr_processor import BatchOCRProcessor
        
        while self.running:
            try:
                # Get next OCR job
                try:
                    job = self.ocr_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                video_path = job['video_path']
                crops_dir = job['crops_dir']
                camera_id = job['camera_id']
                
                # When OCR was deferred (debug flow), ocr is set by callback before jobs are queued
                if self.ocr is None:
                    self.ocr_queue.put(job)
                    time.sleep(0.2)
                    continue
                
                # Wait for any ongoing video processing to complete
                # OCR has priority, but we should let current video processing finish gracefully
                wait_start = time.time()
                with self.state_lock:
                    video_active = self.processing_video
                
                while video_active and self.running:
                    elapsed = time.time() - wait_start
                    if elapsed > 1.0:  # Log every second while waiting
                        log.info(f"[ProcessingQueueManager] OCR queued, waiting for video processing to complete: {Path(crops_dir).name} (waited {elapsed:.1f}s)")
                        wait_start = time.time()
                    time.sleep(0.1)
                    
                    # Re-check video status
                    with self.state_lock:
                        video_active = self.processing_video
                
                if not self.running:
                    # Put job back in queue if shutting down
                    self.ocr_queue.put(job)
                    break
                
                log.info(f"[ProcessingQueueManager] Starting OCR processing: {Path(crops_dir).name}")
                
                # Reset stop request so this job can run (user may have clicked Stop on previous run)
                with self.state_lock:
                    self.ocr_stop_requested = False
                
                # Acquire GPU lock for OCR processing (OCR has priority)
                with self.gpu_lock:
                    # Set OCR processing flag BEFORE starting (so video worker knows to wait)
                    with self.state_lock:
                        self.processing_ocr = True
                    try:
                        # Initialize batch OCR processor
                        batch_processor = BatchOCRProcessor(self.ocr, self.preprocessor)
                        
                        # Process all crops (stops mid-run if request_ocr_stop() was called)
                        ocr_results = batch_processor.process_crops_directory(
                            crops_dir, should_stop=self._should_stop_ocr
                        )
                        
                        # Match OCR results to detections
                        combined_results = batch_processor.match_ocr_to_detections(crops_dir, ocr_results)
                        
                        # Cleanup GPU memory after OCR
                        self._cleanup_gpu_memory()
                        
                        log.info(f"[ProcessingQueueManager] OCR processing complete: {Path(crops_dir).name}")
                        log.info(f"  - Processed {len(combined_results)} crops")
                        
                        self.stats['ocr_jobs_processed'] += 1
                        
                        ocr_cb = job.get('on_ocr_complete') or self.on_ocr_complete
                        if ocr_cb:
                            try:
                                ocr_cb(video_path, crops_dir, combined_results)
                            except Exception as e:
                                log.info(f"[ProcessingQueueManager] Error in OCR completion callback: {e}")
                    
                    except Exception as e:
                        log.info(f"[ProcessingQueueManager] Error processing OCR {crops_dir}: {e}")
                        import traceback
                        traceback.print_exc()
                        self.stats['errors'] += 1
                        self._cleanup_gpu_memory()
                    
                    finally:
                        with self.state_lock:
                            self.processing_ocr = False
                
                self.ocr_queue.task_done()
                
            except Exception as e:
                log.info(f"[ProcessingQueueManager] Error in OCR worker thread: {e}")
                import traceback
                traceback.print_exc()
                self.stats['errors'] += 1
                time.sleep(1)  # Prevent tight error loop
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory after operations."""
        if not TORCH_AVAILABLE:
            return
        
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        with self.state_lock:
            status = {
                'processing_video': self.processing_video,
                'processing_ocr': self.processing_ocr,
                'video_queue_size': self.video_queue.qsize(),
                'ocr_queue_size': self.ocr_queue.qsize(),
                'stats': self.stats.copy()
            }
        if self.defer_ocr:
            with self._pending_lock:
                status['defer_ocr'] = True
                status['pending_ocr_jobs'] = len(self.pending_ocr_jobs)
                status['recording_stopped'] = self.recording_stopped
        return status
    
    def stop(self):
        """Stop the processing queue manager."""
        log.info(f"[ProcessingQueueManager] Stopping...")
        self.running = False
        
        # Wait for queues to empty (with timeout)
        timeout = 30
        start_time = time.time()
        
        with self.state_lock:
            processing_video = self.processing_video
            processing_ocr = self.processing_ocr
        
        while (self.video_queue.qsize() > 0 or self.ocr_queue.qsize() > 0 or 
               processing_video or processing_ocr):
            if time.time() - start_time > timeout:
                log.info(f"[ProcessingQueueManager] Timeout waiting for queues to empty")
                break
            time.sleep(0.5)
            
            with self.state_lock:
                processing_video = self.processing_video
                processing_ocr = self.processing_ocr
        
        log.info(f"[ProcessingQueueManager] Stopped. Final stats: {self.stats}")
