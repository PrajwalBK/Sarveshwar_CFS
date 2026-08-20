"""
CSV Logger with Daily Rotation

Logs trailer detection events to CSV files with automatic daily rotation.
Supports upload of rotated files via uploader module.
"""

import csv
import os
from datetime import datetime, date
from typing import Dict, Optional
from pathlib import Path
import threading


class CSVLogger:
    """
    CSV logger with daily rotation.
    
    Writes events to out/events_YYYYMMDD.csv and rotates daily.
    On rotation, yesterday's file can be uploaded via uploader.
    """
    
    def __init__(self, output_dir: str = "out", uploader=None):
        """
        Initialize CSV logger.
        
        Args:
            output_dir: Base output directory
            uploader: Optional uploader instance for rotating files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.uploader = uploader
        self.current_date = date.today()
        self.current_file = None
        self.writer = None
        self.lock = threading.Lock()
        
        # CSV columns
        self.columns = [
            'ts_iso', 'camera_id', 'track_id', 'bbox', 'text', 'conf',
            'x_world', 'y_world', 'spot', 'method'
        ]
        
        self._open_file()
    
    def _get_filename(self, target_date: date) -> Path:
        """Get CSV filename for a given date."""
        date_str = target_date.strftime('%Y%m%d')
        return self.output_dir / f"events_{date_str}.csv"
    
    def _open_file(self):
        """Open CSV file for current date."""
        filename = self._get_filename(self.current_date)
        
        # Check if file exists to determine if we need header
        file_exists = filename.exists()
        
        self.current_file = open(filename, 'a', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.current_file, fieldnames=self.columns)
        
        if not file_exists:
            self.writer.writeheader()
            self.current_file.flush()
    
    def _rotate_if_needed(self):
        """Rotate to new file if date changed."""
        today = date.today()
        if today != self.current_date:
            # Close old file
            old_file = self.current_file
            old_filename = self._get_filename(self.current_date)
            
            if old_file:
                old_file.close()
            
            # Upload yesterday's file if uploader available
            if self.uploader and old_filename.exists():
                self.uploader.queue_upload(str(old_filename), 'csv')
            
            # Open new file
            self.current_date = today
            self._open_file()
    
    def log(self, event: Dict):
        """
        Log an event to CSV.
        
        Args:
            event: Event dict with keys matching CSV columns
        """
        with self.lock:
            self._rotate_if_needed()
            
            # Ensure all columns are present
            row = {col: event.get(col, '') for col in self.columns}
            
            # Format bbox as string
            if 'bbox' in row and isinstance(row['bbox'], list):
                row['bbox'] = ','.join(map(str, row['bbox']))
            
            self.writer.writerow(row)
            self.current_file.flush()
    
    def close(self):
        """Close current file and upload if needed."""
        with self.lock:
            if self.current_file:
                self.current_file.close()
                self.current_file = None
            
            # Upload current file if uploader available
            if self.uploader:
                filename = self._get_filename(self.current_date)
                if filename.exists():
                    self.uploader.queue_upload(str(filename), 'csv')
    
    def get_today_filename(self) -> str:
        """Get today's CSV filename."""
        return str(self._get_filename(date.today()))


