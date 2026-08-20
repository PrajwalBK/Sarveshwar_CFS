"""
Test / Demo Script for StackerVision Pipeline

Processes an input video file or webcam stream through StackerVisionPipeline,
displays/logs real-time state machine transitions and tier estimates,
and saves the annotated output video.

Usage:
  python tools/test_stacker_vision.py --video path/to/input.mp4 --output out/stacker_demo.mp4
"""

import argparse
import sys
import os
import time
import cv2
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipelines.stacker_vision_pipeline import StackerVisionPipeline
from app.ai.detector_yolov8 import YOLOv8Detector


def main():
    parser = argparse.ArgumentParser(description="Test StackerVision Pipeline on video")
    parser.add_argument("--video", type=str, default="IMG_1399.MOV", help="Path to input video file or camera index (0)")
    parser.add_argument("--output", type=str, default="out/stacker_demo_annotated.mp4", help="Path to output annotated video")
    parser.add_argument("--model", type=str, default="yolov8m.pt", help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--show", action="store_true", help="Show live preview window")
    args = parser.parse_args()

    video_source = args.video
    if video_source.isdigit():
        video_source = int(video_source)

    print(f"[TestStackerVision] Opening video source: {video_source}")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[TestStackerVision] Resolution: {width}x{height} @ {fps:.1f} FPS (Total frames: {total_frames})")

    # Initialize Detector
    detector = None
    try:
        print(f"[TestStackerVision] Loading detector model: {args.model}...")
        detector = YOLOv8Detector(model_name=args.model, conf_threshold=args.conf)
    except Exception as e:
        print(f"[TestStackerVision] Warning: Detector failed to load: {e}. Running in heuristic mode.")

    # Initialize StackerVisionPipeline
    pipeline = StackerVisionPipeline(
        camera_id="stacker-test-01",
        detector=detector,
        config={
            "tier_calibration": {
                "method": "spreader_reference",
                "spreader_real_height_m": 0.45,
                "container_real_height_m": 2.591,
            },
            "state_machine": {
                "alignment_iou_thresh": 0.2,
                "co_motion_min_frames": 3,
            }
        }
    )

    # Prepare VideoWriter
    out_writer = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        print(f"[TestStackerVision] Saving annotated video to: {out_path}")

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            pipeline.process_frame(camera_id="stacker-test-01", frame=frame, frame_count=frame_count)

            annotated_frame = pipeline.last_annotated_frame if pipeline.last_annotated_frame is not None else frame

            if out_writer:
                out_writer.write(annotated_frame)

            if args.show:
                cv2.imshow("StackerVision Real-Time Status", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[TestStackerVision] User interrupted preview.")
                    break

            if frame_count % 30 == 0:
                status = pipeline.get_status()
                print(f"Frame [{frame_count}/{total_frames}] -> State: {status['operational_state']} | Attached: {status['container_attached']} | Tier: {status['tier']} | FPS: {status['fps']}")

    finally:
        cap.release()
        if out_writer:
            out_writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    avg_fps = frame_count / max(0.001, elapsed)
    print(f"\n[TestStackerVision] Done! Processed {frame_count} frames in {elapsed:.2f}s (Avg FPS: {avg_fps:.1f})")
    if args.output:
        print(f"[TestStackerVision] Output video saved to: {args.output}")


if __name__ == "__main__":
    main()
