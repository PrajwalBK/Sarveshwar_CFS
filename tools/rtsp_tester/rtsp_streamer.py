"""
Low-Latency RTSP Streamer & Capture Engine
Designed for Direct CP Plus PoE IP Camera Testing (Offline / No NVR / Windows & Jetson Ready)
"""

import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np


class StreamStatus:
    DISCONNECTED = "Disconnected"
    CONNECTING = "Connecting..."
    CONNECTED = "Connected"
    RECONNECTING = "Reconnecting..."
    ERROR = "Connection Error"


class RTSPStreamer:
    """
    Decoupled, multi-threaded RTSP stream consumer optimized for sub-100ms latency.
    
    Architecture:
      - Capture Thread: Pulls frames from camera socket as fast as camera delivers (over TCP/GStreamer).
      - Atomic Single-Frame Slot: Keeps ONLY the newest frame; automatically drops backlog frames.
      - Display/Consumer Thread: Reads the newest frame on demand without blocking or queue accumulation.
      - Recorder: Writes received frames locally to MP4 in a non-blocking queue.
    """

    def __init__(
        self,
        rtsp_url: str,
        backend: str = "opencv_tcp",
        auto_reconnect: bool = True,
        reconnect_delay_sec: float = 3.0,
        output_dir: Optional[Path] = None,
    ):
        self.rtsp_url = rtsp_url.strip()
        self.backend = backend.lower()
        self.auto_reconnect = auto_reconnect
        self.reconnect_delay_sec = reconnect_delay_sec

        # Directory structure
        base_dir = output_dir or Path("output")
        self.captures_dir = base_dir / "captures"
        self.recordings_dir = base_dir / "recordings"
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

        # Threading & Control Flags
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None

        # Frame storage (Single-Slot Atomic Buffer)
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_ts: float = 0.0
        self._frame_width: int = 0
        self._frame_height: int = 0
        self._camera_fps: float = 0.0

        # Diagnostics & Metrics
        self.status: str = StreamStatus.DISCONNECTED
        self.error_message: str = ""
        self.frames_received: int = 0
        self.frames_consumed: int = 0
        self.frames_dropped: int = 0
        self.reconnect_count: int = 0
        self.rx_fps: float = 0.0
        self.render_fps: float = 0.0
        self.estimated_latency_ms: float = 0.0

        # Performance measurement rolling windows
        self._rx_timestamps: list = []
        self._render_timestamps: list = []

        # Local Recording
        self._is_recording: bool = False
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._recording_path: Optional[Path] = None
        self._recording_frame_count: int = 0

    @staticmethod
    def is_gstreamer_supported() -> bool:
        """Check if OpenCV was compiled with GStreamer support."""
        try:
            build_info = cv2.getBuildInformation()
            return "GStreamer:" in build_info and "YES" in build_info.split("GStreamer:")[1].split("\n")[0]
        except Exception:
            return False

    def _build_gstreamer_pipeline(self) -> str:
        """Construct ultra-low-latency GStreamer pipeline with zero buffering."""
        if self.backend == "jetson":
            # NVIDIA Jetson hardware-accelerated NVMM pipeline
            return (
                f"rtspsrc location=\"{self.rtsp_url}\" latency=0 drop-on-latency=true protocols=tcp ! "
                f"rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 ! "
                f"nvvidconv ! video/x-raw, format=BGRx ! "
                f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false"
            )
        elif self.backend in ("gstreamer_h265", "gstreamer_hevc"):
            # GStreamer H.265 / HEVC decode
            return (
                f"rtspsrc location=\"{self.rtsp_url}\" latency=0 drop-on-latency=true protocols=tcp ! "
                f"rtph265depay ! h265parse ! avdec_h265 ! "
                f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false"
            )
        else:
            # GStreamer H.264 CPU decode (Windows/Linux)
            return (
                f"rtspsrc location=\"{self.rtsp_url}\" latency=0 drop-on-latency=true protocols=tcp ! "
                f"rtph264depay ! h264parse ! avdec_h264 ! "
                f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false"
            )

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Open RTSP stream using the selected backend with zero-latency flags."""
        # Ultra low-latency environment flags for FFmpeg backend
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0|"
            "analyzeduration;0|probesize;32|reorder_queue_size;0|sync;ext|framedrop;1"
        )

        if self.backend in ("gstreamer", "jetson", "gstreamer_h265"):
            pipeline = self._build_gstreamer_pipeline()
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            # Standard OpenCV FFmpeg with TCP transport
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            return None

        # Configure OpenCV buffer size (request single frame buffer)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        return cap

    def start(self) -> bool:
        """Start the background RTSP capture thread."""
        if self._capture_thread and self._capture_thread.is_alive():
            return True

        self._stop_event.clear()
        self.status = StreamStatus.CONNECTING
        self.error_message = ""

        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="RTSPWorkerThread", daemon=True
        )
        self._capture_thread.start()
        return True

    def stop(self):
        """Stop streaming and release all resources."""
        self._stop_event.set()
        self.stop_recording()

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)

        with self._lock:
            self._latest_frame = None
            self.status = StreamStatus.DISCONNECTED

    def _capture_loop(self):
        """Dedicated thread continuously pulling latest frames from RTSP socket."""
        while not self._stop_event.is_set():
            cap = None
            try:
                self.status = (
                    StreamStatus.CONNECTING
                    if self.reconnect_count == 0
                    else StreamStatus.RECONNECTING
                )
                t_conn_start = time.monotonic()
                cap = self._open_capture()

                if cap is None or not cap.isOpened():
                    raise ConnectionError(f"Failed to connect to RTSP stream: {self.rtsp_url}")

                # Connected successfully
                self.status = StreamStatus.CONNECTED
                self.error_message = ""
                self.reconnect_count = 0
                conn_latency = (time.monotonic() - t_conn_start) * 1000

                # Read stream properties
                self._frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
                self._frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
                self._camera_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0

                frame_grab_time = time.monotonic()

                while not self._stop_event.is_set():
                    t_grab_start = time.monotonic()
                    ret, frame = cap.read()

                    if not ret or frame is None:
                        raise ConnectionResetError("RTSP stream read returned empty frame (packet drop or connection lost).")

                    t_grab_end = time.monotonic()
                    frame_ingest_latency = (t_grab_end - t_grab_start) * 1000

                    with self._lock:
                        if self._latest_frame is not None:
                            # Previous frame was not read by UI before this one arrived -> counted as dropped
                            self.frames_dropped += 1

                        self._latest_frame = frame
                        self._latest_frame_ts = t_grab_end
                        self.frames_received += 1
                        self.estimated_latency_ms = frame_ingest_latency

                        # Track received FPS
                        now = time.monotonic()
                        self._rx_timestamps.append(now)
                        # Keep 1-second rolling window
                        while self._rx_timestamps and now - self._rx_timestamps[0] > 1.0:
                            self._rx_timestamps.pop(0)
                        self.rx_fps = len(self._rx_timestamps)

                        # Write to local recording if active
                        if self._is_recording and self._video_writer is not None:
                            try:
                                self._video_writer.write(frame)
                                self._recording_frame_count += 1
                            except Exception as rec_err:
                                print(f"[Recorder Error] {rec_err}")

            except Exception as e:
                self.status = StreamStatus.ERROR
                self.error_message = str(e)
                with self._lock:
                    self._latest_frame = None

                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

                if not self.auto_reconnect or self._stop_event.is_set():
                    break

                self.reconnect_count += 1
                self.status = StreamStatus.RECONNECTING
                # Wait before reconnecting
                time.sleep(self.reconnect_delay_sec)

            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """
        Retrieve the newest available frame without blocking.
        Returns: (success, frame_bgr, metrics_dict)
        """
        with self._lock:
            frame = self._latest_frame
            # Consume the frame so we don't count duplicate reads
            self._latest_frame = None

            if frame is not None:
                self.frames_consumed += 1
                now = time.monotonic()
                self._render_timestamps.append(now)
                while self._render_timestamps and now - self._render_timestamps[0] > 1.0:
                    self._render_timestamps.pop(0)
                self.render_fps = len(self._render_timestamps)

            metrics = {
                "status": self.status,
                "error": self.error_message,
                "resolution": f"{self._frame_width}x{self._frame_height}" if self._frame_width else "N/A",
                "camera_fps": self._camera_fps,
                "rx_fps": self.rx_fps,
                "render_fps": self.render_fps,
                "latency_ms": self.estimated_latency_ms,
                "frames_received": self.frames_received,
                "frames_consumed": self.frames_consumed,
                "frames_dropped": self.frames_dropped,
                "reconnect_count": self.reconnect_count,
                "is_recording": self._is_recording,
                "recording_frames": self._recording_frame_count,
            }

            return (frame is not None), frame, metrics

    def capture_snapshot(self) -> Tuple[bool, str]:
        """Save the current frame locally with a timestamp."""
        with self._lock:
            frame = self._latest_frame
        
        if frame is None:
            return False, "No active frame available to snapshot."

        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            filename = f"snapshot_{ts}.jpg"
            filepath = self.captures_dir / filename
            cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return True, str(filepath)
        except Exception as e:
            return False, f"Failed to save snapshot: {e}"

    def start_recording(self) -> Tuple[bool, str]:
        """Start saving incoming video frames to a local MP4 file."""
        if self._is_recording:
            return False, "Recording is already active."

        if self._frame_width <= 0 or self._frame_height <= 0:
            return False, "Cannot start recording: No video stream connected yet."

        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{ts}.mp4"
            filepath = self.recordings_dir / filename

            # Use MP4V / H264 codec
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = self._camera_fps if (10.0 <= self._camera_fps <= 60.0) else 25.0

            writer = cv2.VideoWriter(
                str(filepath), fourcc, fps, (self._frame_width, self._frame_height)
            )

            if not writer.isOpened():
                return False, "Failed to initialize VideoWriter."

            with self._lock:
                self._video_writer = writer
                self._recording_path = filepath
                self._recording_frame_count = 0
                self._is_recording = True

            return True, str(filepath)
        except Exception as e:
            return False, f"Failed to start recording: {e}"

    def stop_recording(self) -> Tuple[bool, str]:
        """Stop local video recording."""
        if not self._is_recording:
            return False, "Recording is not active."

        with self._lock:
            self._is_recording = False
            writer = self._video_writer
            self._video_writer = None
            path = self._recording_path
            count = self._recording_frame_count

        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass

        return True, f"Saved {count} frames to {path}"
