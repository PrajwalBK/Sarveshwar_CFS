"""
Extract Frames for Training Data

This script processes a video file and extracts full frames every second,
saving them to training_data/train/ folder for OCR training.

Usage:
    python tools/extract_training_crops.py --video path/to/video.mp4
    python tools/extract_training_crops.py --video path/to/video.mp4 --output training_data/train
    python tools/extract_training_crops.py --video path/to/video.mp4 --interval 2.0  # Every 2 seconds
"""

import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict


class FrameExtractor:
    """Extract full frames from video for training."""
    
    def __init__(self):
        """Initialize extractor."""
        print("[Extractor] Frame extractor initialized (no detector/OCR needed)")
        
    def extract_frames(self, video_path: str, output_dir: str = "training_data/train",
                      interval_seconds: float = 1.0) -> Dict:
        """
        Extract full frames from video.
        
        Args:
            video_path: Path to input video file
            output_dir: Output directory for frames
            interval_seconds: Extract frames every N seconds
            
        Returns:
            Dictionary with extraction statistics
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"[Extractor] Output directory: {output_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"[Extractor] Video properties:")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Duration: {duration:.1f} seconds")
        print(f"  Extraction interval: {interval_seconds} seconds")
        
        # Calculate frame interval
        frame_interval = max(1, int(fps * interval_seconds))
        print(f"  Frame interval: {frame_interval} frames")
        
        # Statistics
        stats = {
            'frames_processed': 0,
            'frames_extracted': 0,
            'frames_saved': 0
        }
        
        # Check for existing image files and find the highest number
        # This allows appending new images instead of overwriting
        existing_images = list(output_path.glob("img_*.jpg"))
        if existing_images:
            # Extract numbers from existing filenames (img_00001.jpg -> 1)
            existing_numbers = []
            for img_path in existing_images:
                try:
                    # Extract number from filename like "img_00001.jpg"
                    num_str = img_path.stem.split('_')[1]  # "00001"
                    num = int(num_str)
                    existing_numbers.append(num)
                except (ValueError, IndexError):
                    # Skip files that don't match the pattern
                    continue
            
            # Start from the next number after the highest existing
            if existing_numbers:
                frame_counter = max(existing_numbers)
                print(f"[Extractor] Found {len(existing_images)} existing images. Starting from img_{frame_counter + 1:05d}.jpg")
            else:
                frame_counter = 0
        else:
            frame_counter = 0
            print(f"[Extractor] No existing images found. Starting from img_00001.jpg")
        
        frame_count = 0
        
        print("\n[Extractor] Starting extraction...")
        print("-" * 60)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extract frames every N frames (every second)
                if frame_count % frame_interval == 0:
                    stats['frames_extracted'] += 1
                    
                    # Generate filename (increment counter first)
                    frame_counter += 1
                    filename = f"img_{frame_counter:05d}.jpg"
                    filepath = output_path / filename
                    
                    # Skip if file already exists (shouldn't happen with proper numbering, but safety check)
                    if filepath.exists():
                        print(f"  Frame {frame_count:05d}: Skipping {filename} (already exists)")
                        continue
                    
                    # Save full frame
                    cv2.imwrite(str(filepath), frame)
                    stats['frames_saved'] += 1
                    
                    print(f"  Frame {frame_count:05d}: Saved {filename}")
                
                frame_count += 1
                stats['frames_processed'] += 1
                
                # Progress update every 100 frames
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    print(f"[Extractor] Progress: {frame_count}/{total_frames} frames ({progress:.1f}%) - "
                          f"{stats['frames_saved']} frames saved")
        
        finally:
            cap.release()
        
        print("-" * 60)
        print(f"[Extractor] Extraction complete!")
        print(f"  Frames processed: {stats['frames_processed']}")
        print(f"  Frames extracted: {stats['frames_extracted']}")
        print(f"  Frames saved: {stats['frames_saved']}")
        print(f"  Output directory: {output_path}")
        
        return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract full frames from video for OCR training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract frames every second (default)
  python tools/extract_training_crops.py --video test_video.mp4
  
  # Extract frames every 2 seconds
  python tools/extract_training_crops.py --video test_video.mp4 --interval 2.0
  
  # Custom output directory
  python tools/extract_training_crops.py --video test_video.mp4 --output my_training_data/train
        """
    )
    
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="training_data/train",
        help="Output directory for crops (default: training_data/train)"
    )
    
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Extract crops every N seconds (default: 1.0)"
    )
    
    
    args = parser.parse_args()
    
    try:
        # Initialize extractor
        extractor = FrameExtractor()
        
        # Extract frames
        stats = extractor.extract_frames(
            video_path=args.video,
            output_dir=args.output,
            interval_seconds=args.interval
        )
        
        print("\n✓ Extraction completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n[Extractor] Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[Extractor] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

