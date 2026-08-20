"""
Video Recorder with GPS Logging and Auto-Chunking

Records video from camera and logs GPS coordinates every second.
Automatically splits recordings into 45-second chunks.
GPS data is stored in a JSON file for later matching during video processing.
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import threading
import time
from typing import Optional, Dict, Callable

from app.app_logger import get_logger

log = get_logger(__name__)


class VideoRecorder:
    """
    Video recorder that captures video and GPS data simultaneously.
    Supports automatic 45-second chunking for continuous recording.
    """
    
    def __init__(self, output_dir: str = "out/recordings", gps_sensor=None, 
                 chunk_duration_seconds: float = 45.0, on_chunk_saved: Optional[Callable] = None):
        """
        Initialize video recorder.
        
        Args:
            output_dir: Directory to save recorded videos
            gps_sensor: Optional GPSSensor instance
            chunk_duration_seconds: Duration of each video chunk in seconds (default: 45.0)
            on_chunk_saved: Optional callback(video_path, gps_log_path) when a chunk is saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gps_sensor = gps_sensor
        self.chunk_duration_seconds = chunk_duration_seconds
        self.on_chunk_saved = on_chunk_saved
        
        self.recording = False
        self.video_writer = None
        self.video_path = None
        self.gps_log_path = None
        self.gps_log = {}  # timestamp -> {'lat': float, 'lon': float, 'heading': float (optional), 'speed': float (optional)}
        self.lock = threading.Lock()
        
        self.camera_id = None
        self.frame_width = None
        self.frame_height = None
        self.fps = 30
        
        # Auto-chunking state
        self.chunk_start_time = None
        self.frame_count = 0
        self.chunk_index = 0
        self.current_chunk_gps_log = {}  # GPS log for current chunk
        
        # GPS logging thread
        self.gps_logging_thread = None
        self.gps_logging_running = False
    
    def start_recording(self, camera_id: str, frame_width: int, frame_height: int, fps: int = 30):
        """
        Start recording video and GPS data with automatic chunking.
        
        Args:
            camera_id: Camera identifier
            frame_width: Video frame width
            frame_height: Video frame height
            fps: Video frame rate
        """
        if self.recording:
            log.info(f"[VideoRecorder] Already recording, stop current recording first")
            return False
        
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        
        # Reset chunking state
        self.chunk_index = 0
        self.frame_count = 0
        self.chunk_start_time = time.time()
        self.current_chunk_gps_log = {}
        
        # Start first chunk
        self._start_new_chunk()
        
        self.recording = True
        
        # Start GPS logging thread
        if self.gps_sensor:
            self.gps_logging_running = True
            self.gps_logging_thread = threading.Thread(
                target=self._gps_logging_loop,
                daemon=True
            )
            self.gps_logging_thread.start()
            log.info(f"[VideoRecorder] Started GPS logging thread")
        else:
            log.info("[VideoRecorder] GPS not connected - GPS log file will not be stored for this recording")
        
        log.info(f"[VideoRecorder] Started recording with {self.chunk_duration_seconds}s auto-chunking")
        return True
    
    def _start_new_chunk(self):
        """Start a new video chunk."""
        # Close previous chunk if exists
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        # Save previous chunk if it exists
        if self.video_path is not None and Path(self.video_path).exists():
            self._save_chunk()
        
        # Generate filename for new chunk
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_filename = f"{self.camera_id}_{timestamp}_chunk{self.chunk_index:04d}.mp4"
        gps_filename = f"{self.camera_id}_{timestamp}_chunk{self.chunk_index:04d}_gps.json"
        
        self.video_path = self.output_dir / video_filename
        self.gps_log_path = self.output_dir / gps_filename
        
        # Initialize video writer for new chunk
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        if not self.video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for chunk {self.chunk_index}")
        
        # Reset chunk state
        self.chunk_start_time = time.time()
        self.frame_count = 0
        self.current_chunk_gps_log = {}
        
        log.info(f"[VideoRecorder] Started chunk {self.chunk_index}: {self.video_path.name}")
    
    def _save_chunk(self):
        """Save current chunk and trigger callback."""
        if self.video_path is None or not Path(self.video_path).exists():
            return
        
        # Save GPS log for this chunk
        if self.current_chunk_gps_log and self.gps_log_path:
            try:
                with open(self.gps_log_path, 'w') as f:
                    json.dump(self.current_chunk_gps_log, f, indent=2)
                log.info(f"[VideoRecorder] Saved GPS log for chunk {self.chunk_index}: {self.gps_log_path.name} ({len(self.current_chunk_gps_log)} entries)")
            except Exception as e:
                log.warning("Error saving GPS log for chunk %s: %s", self.chunk_index, e)
        
        video_path = str(self.video_path)
        gps_log_path = str(self.gps_log_path) if self.gps_log_path else None
        
        log.info(f"[VideoRecorder] Saved chunk {self.chunk_index}: {Path(video_path).name} ({self.frame_count} frames, {time.time() - self.chunk_start_time:.1f}s)")
        
        # Trigger callback if provided
        if self.on_chunk_saved:
            try:
                self.on_chunk_saved(video_path, gps_log_path)
            except Exception as e:
                log.warning("Error in chunk saved callback: %s", e)
        
        self.chunk_index += 1
    
    def _gps_logging_loop(self):
        """
        Background thread that logs GPS coordinates every second.
        Logs to both global log and current chunk log.
        """
        while self.gps_logging_running:
            if self.gps_sensor:
                gps_data = self.gps_sensor.get_current_gps()
                if gps_data:
                    timestamp = datetime.utcnow()
                    timestamp_str = timestamp.isoformat()
                    
                    log_entry = {
                        'lat': gps_data['lat'],
                        'lon': gps_data['lon'],
                        'timestamp': timestamp_str
                    }
                    # Include heading and speed if available
                    if 'heading' in gps_data:
                        log_entry['heading'] = gps_data['heading']
                    if 'speed' in gps_data:
                        log_entry['speed'] = gps_data['speed']
                    
                    with self.lock:
                        # Add to global log
                        self.gps_log[timestamp_str] = log_entry
                        # Add to current chunk log
                        self.current_chunk_gps_log[timestamp_str] = log_entry
                    
                    # Log every 10 seconds for debugging
                    if len(self.gps_log) % 10 == 0:
                        log.info(f"[VideoRecorder] GPS logged: ({gps_data['lat']:.6f}, {gps_data['lon']:.6f}) - Total entries: {len(self.gps_log)}")
            
            # Sleep for 0.1 seconds (check 10 times per second)
            time.sleep(0.1)
    
    def write_frame(self, frame: np.ndarray):
        """
        Write a frame to the video file.
        Automatically starts a new chunk after chunk_duration_seconds.
        
        Args:
            frame: BGR image frame
            
        Returns:
            True if frame was written, False otherwise
        """
        if not self.recording:
            return False
        
        # Check if we need to start a new chunk
        elapsed_time = time.time() - self.chunk_start_time
        if elapsed_time >= self.chunk_duration_seconds:
            # Save current chunk and start new one
            self._start_new_chunk()
        
        if self.video_writer is None:
            return False
        
        # Resize frame if needed
        if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        self.video_writer.write(frame)
        self.frame_count += 1
        return True
    
    def stop_recording(self):
        """
        Stop recording and save final chunk.
        
        Returns:
            Tuple of (last_video_path, last_gps_log_path) or (None, None) if not recording
        """
        if not self.recording:
            return None, None
        
        self.recording = False
        self.gps_logging_running = False
        
        # Wait for GPS logging thread to finish
        if self.gps_logging_thread:
            self.gps_logging_thread.join(timeout=2.0)
        
        # Save final chunk
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        # Save final chunk
        if self.video_path is not None:
            self._save_chunk()
        
        video_path = str(self.video_path) if self.video_path and Path(self.video_path).exists() else None
        gps_log_path = str(self.gps_log_path) if self.gps_log_path and Path(self.gps_log_path).exists() else None
        
        log.info(f"[VideoRecorder] Stopped recording. Total chunks: {self.chunk_index}")
        
        # Reset paths
        self.video_path = None
        self.gps_log_path = None
        
        return video_path, gps_log_path
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.recording
    
    def get_gps_log(self) -> Dict:
        """Get current GPS log (for testing/debugging)."""
        with self.lock:
            return self.gps_log.copy()
