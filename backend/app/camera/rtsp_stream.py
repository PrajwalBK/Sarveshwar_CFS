"""OpenCV boundary. Native libraries are imported only when capture is requested."""
import os
from pathlib import Path


class OpenCVStream:
    def __init__(self, source, source_type, settings):
        # Suppress native decoder logs that can include credential-bearing URLs.
        os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL', '-8')
        os.environ.setdefault('OPENCV_LOG_LEVEL', 'SILENT')
        os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp')
        import cv2
        if hasattr(cv2, 'setLogLevel'):
            cv2.setLogLevel(0)
        self.cv2 = cv2
        if source_type == 'rtsp':
            if not source.lower().startswith(('rtsp://', 'rtsps://')):
                raise ValueError('Configured camera source is not RTSP')
            open_timeout = int(getattr(settings, 'open_timeout_ms', 5000))
            read_timeout = int(getattr(settings, 'read_timeout_ms', 3000))
            timeout_us = open_timeout * 1000
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = f'rtsp_transport;tcp|stimeout;{timeout_us}|timeout;{timeout_us}'
            clean_source = source.replace('%40', '@')
            cap_params = []
            if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC'):
                cap_params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout])
            if hasattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC'):
                cap_params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout])
            self.cap = cv2.VideoCapture(clean_source, cv2.CAP_FFMPEG, cap_params) if cap_params else cv2.VideoCapture(clean_source, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                # Fallback to UDP transport if TCP transport is disabled on camera
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = f'rtsp_transport;udp|stimeout;{timeout_us}|timeout;{timeout_us}'
                self.cap.release()
                self.cap = cv2.VideoCapture(clean_source, cv2.CAP_FFMPEG, cap_params) if cap_params else cv2.VideoCapture(clean_source, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            if not Path(source).is_file():
                raise FileNotFoundError('Configured replay file is unavailable')
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap.release()
            raise ConnectionError('Capture unavailable')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 0 < fps < 240 else 25.0
        self.frame_count = max(0, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))) if source_type == 'file' else 0

    def read(self):
        return self.cap.read()

    def close(self):
        self.cap.release()
