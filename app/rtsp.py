"""
Camera Stream Capture Module

Handles RTSP/USB camera capture using OpenCV.
For Jetson production, use GStreamer pipeline for NVDEC hardware acceleration.
"""

import os
# Suppress OpenCV warnings at module level
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

import cv2
import numpy as np
from typing import Optional, Generator, Tuple


def build_gstreamer_pipeline(rtsp_url: str, width: int, height: int, fps: int = 30) -> str:
    """
    Build GStreamer pipeline for Jetson NVDEC hardware decoding.
    
    Args:
        rtsp_url: RTSP stream URL
        width: Frame width
        height: Frame height
        fps: Frame rate
        
    Returns:
        GStreamer pipeline string
    """
    # GStreamer pipeline for RTSP with NVDEC on Jetson
    pipeline = (
        f"rtspsrc location={rtsp_url} latency=0 ! "
        "rtph264depay ! "
        "h264parse ! "
        "nvv4l2decoder ! "
        "nvvidconv ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )
    return pipeline


def open_stream(rtsp_url: str, width: int, height: int, fps_cap: int = 30, use_gstreamer: bool = False) -> Optional[cv2.VideoCapture]:
    """
    Open camera stream (RTSP or USB).
    
    Args:
        rtsp_url: RTSP URL or device index (e.g., "0" for USB camera)
        width: Expected frame width
        height: Expected frame height
        fps_cap: Frame rate cap
        use_gstreamer: Use GStreamer pipeline (recommended for Jetson)
        
    Returns:
        OpenCV VideoCapture object or None if failed
    """
    if use_gstreamer and rtsp_url.startswith('rtsp://'):
        # Use GStreamer pipeline for RTSP on Jetson
        pipeline = build_gstreamer_pipeline(rtsp_url, width, height, fps_cap)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    elif rtsp_url.isdigit() or rtsp_url == '0':
        # USB camera. Force V4L2 backend on Linux — otherwise OpenCV picks
        # GStreamer's v4l2src, which negotiates YUYV and fails on cameras
        # like the LifeCam HD-6000 that only support 720p30 over MJPG.
        device_idx = int(rtsp_url)
        backend = cv2.CAP_V4L2 if hasattr(cv2, 'CAP_V4L2') else cv2.CAP_ANY
        cap = cv2.VideoCapture(device_idx, backend)
        if cap.isOpened():
            # FourCC must be set BEFORE width/height/fps so the driver
            # negotiates the right pixel format for the requested mode.
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except Exception:
                pass
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps_cap)
    else:
        # RTSP with OpenCV (force TCP & low-latency nobuffer)
        if str(rtsp_url).startswith('rtsp://'):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0|"
                "analyzeduration;0|probesize;32|reorder_queue_size;0|sync;ext|framedrop;1"
            )
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
        else:
            cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        # Don't print error - let caller handle it
        # This is expected when checking camera availability
        return None
    
    return cap


def frame_generator(cap: cv2.VideoCapture) -> Generator[Tuple[bool, Optional[np.ndarray]], None, None]:
    """
    Generator that yields frames from a VideoCapture.
    
    Args:
        cap: OpenCV VideoCapture object
        
    Yields:
        Tuple of (success, frame) where frame is BGR image or None
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield ret, frame
    
    cap.release()


def test_stream(rtsp_url: str, width: int, height: int, use_gstreamer: bool = False) -> bool:
    """
    Test if a stream can be opened and read.
    Suppresses OpenCV warnings during testing.
    
    Args:
        rtsp_url: RTSP URL or device index
        width: Expected width
        height: Expected height
        use_gstreamer: Use GStreamer pipeline
        
    Returns:
        True if stream is accessible, False otherwise
    """
    try:
        # Suppress OpenCV warnings using multiple methods
        import os
        import sys
        from contextlib import redirect_stderr
        
        # Method 1: Set OpenCV log level to ERROR (suppress WARN and INFO)
        try:
            cv2.setLogLevel(1)  # 0=VERBOSE, 1=ERROR, 2=WARN, 3=INFO, 4=DEBUG, 5=SILENT
        except:
            pass  # Older OpenCV versions might not have this
        
        # Method 2: Redirect stderr to devnull
        old_stderr = sys.stderr
        try:
            with open(os.devnull, 'w') as devnull:
                sys.stderr = devnull
                try:
                    cap = open_stream(rtsp_url, width, height, use_gstreamer=use_gstreamer)
                    if cap is None:
                        return False
                    
                    ret, frame = cap.read()
                    cap.release()
                    return ret and frame is not None
                finally:
                    sys.stderr = old_stderr
        except:
            sys.stderr = old_stderr
            return False
    except Exception:
        # Any exception means camera is not accessible
        return False
    finally:
        # Restore OpenCV log level
        try:
            cv2.setLogLevel(2)  # Restore to WARN level
        except:
            pass


