"""
Screenshot Rotator with Keep-Last Policy

Manages screenshot rotation per camera with configurable keep-last policy.
Supports upload-on-save or upload-on-rotate.
"""

import os
from pathlib import Path
from typing import Optional
import threading
from datetime import datetime
import cv2
import numpy as np


class MediaRotator:
    """
    Screenshot rotator with keep-last policy per camera.
    
    Maintains a fixed number of recent screenshots per camera,
    deleting older ones when the limit is exceeded.
    """
    
    def __init__(self, output_dir: str = "out/screenshots", keep_last: int = 10, uploader=None):
        """
        Initialize media rotator.
        
        Args:
            output_dir: Base output directory for screenshots
            keep_last: Number of screenshots to keep per camera
            uploader: Optional uploader instance for uploading screenshots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.uploader = uploader
        self.lock = threading.Lock()
    
    def save_frame(self, camera_id: str, frame: np.ndarray, track_id: Optional[int] = None) -> Optional[str]:
        """
        Save a screenshot for a camera.
        
        Args:
            camera_id: Camera identifier
            frame: BGR image frame
            track_id: Optional track ID to include in filename
            
        Returns:
            Path to saved screenshot or None if failed
        """
        with self.lock:
            # Create camera directory
            camera_dir = self.output_dir / camera_id
            camera_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            if track_id is not None:
                filename = f"{timestamp}_track{track_id}.jpg"
            else:
                filename = f"{timestamp}.jpg"
            
            filepath = camera_dir / filename
            
            # Save image
            try:
                cv2.imwrite(str(filepath), frame)
                
                # Upload if uploader available
                if self.uploader:
                    self.uploader.queue_upload(str(filepath), 'screenshot')
                
                # Rotate (keep only last N)
                self._rotate_camera_dir(camera_dir)
                
                return str(filepath)
            except Exception as e:
                print(f"Failed to save screenshot: {e}")
                return None
    
    def _rotate_camera_dir(self, camera_dir: Path):
        """
        Rotate screenshots in a camera directory, keeping only the last N files.
        
        Args:
            camera_dir: Directory containing screenshots for a camera
        """
        # Get all image files sorted by modification time
        image_files = sorted(
            camera_dir.glob("*.jpg"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Delete files beyond keep_last
        for file_to_delete in image_files[self.keep_last:]:
            try:
                file_to_delete.unlink()
            except Exception as e:
                print(f"Failed to delete old screenshot {file_to_delete}: {e}")
    
    def get_camera_screenshots(self, camera_id: str) -> list:
        """
        Get list of screenshot paths for a camera.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            List of screenshot file paths, sorted by modification time (newest first)
        """
        camera_dir = self.output_dir / camera_id
        if not camera_dir.exists():
            return []
        
        image_files = sorted(
            camera_dir.glob("*.jpg"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        return [str(f) for f in image_files]


