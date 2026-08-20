"""
Headless RTSP Stream Tester CLI
For Low-Latency Benchmarking & NVIDIA Jetson Deployment Testing
"""

import sys
import time
import argparse
from pathlib import Path

from rtsp_streamer import RTSPStreamer, StreamStatus


def main():
    parser = argparse.ArgumentParser(description="Low-Latency CP Plus RTSP Stream Tester (CLI)")
    parser.add_argument("--url", "-u", required=True, help="Full RTSP URL (e.g. rtsp://admin:admin@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0)")
    parser.add_argument("--backend", "-b", default="opencv_tcp", choices=["opencv_tcp", "gstreamer", "jetson"], help="RTSP backend engine")
    parser.add_argument("--duration", "-d", type=int, default=30, help="Test duration in seconds (0 = infinite)")
    parser.add_argument("--snapshot", "-s", action="store_true", help="Capture snapshot after connecting")
    parser.add_argument("--record", "-r", action="store_true", help="Record stream during test")
    parser.add_argument("--output", "-o", default="output", help="Output directory")

    args = parser.parse_args()

    out_dir = Path(args.output)
    streamer = RTSPStreamer(
        rtsp_url=args.url,
        backend=args.backend,
        output_dir=out_dir,
    )

    print("=" * 75)
    print("🎥 CP Plus Low-Latency RTSP Stream Tester (CLI)")
    print(f"URL:     {args.url}")
    print(f"Backend: {args.backend}")
    print(f"Output:  {out_dir.resolve()}")
    print("=" * 75)

    streamer.start()
    t_start = time.monotonic()
    has_snapshotted = False

    try:
        while True:
            elapsed = time.monotonic() - t_start
            if args.duration > 0 and elapsed >= args.duration:
                break

            has_frame, frame, metrics = streamer.get_latest_frame()

            if metrics["status"] == StreamStatus.CONNECTED:
                if args.snapshot and not has_snapshotted and has_frame:
                    ok, path = streamer.capture_snapshot()
                    if ok:
                        print(f"\n[SNAPSHOT] Saved: {path}")
                    has_snapshotted = True

                if args.record and not streamer._is_recording:
                    ok, path = streamer.start_recording()
                    if ok:
                        print(f"\n[RECORDING] Started: {path}")

            # Print single-line live telemetry HUD
            sys.stdout.write(
                f"\r[{metrics['status']:<14}] "
                f"Res: {metrics['resolution']:<9} | "
                f"Rx FPS: {metrics['rx_fps']:4.1f} | "
                f"Render FPS: {metrics['render_fps']:4.1f} | "
                f"Ingest: {metrics['latency_ms']:4.1f}ms | "
                f"Rx Frames: {metrics['frames_received']:<5} | "
                f"Dropped: {metrics['frames_dropped']:<4} | "
                f"Reconnects: {metrics['reconnect_count']}"
            )
            sys.stdout.flush()
            time.sleep(0.04)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        if streamer._is_recording:
            streamer.stop_recording()
        streamer.stop()

    print("\n" + "=" * 75)
    print("Test Complete. Telemetry Summary:")
    print(f"Total Frames Received: {streamer.frames_received}")
    print(f"Total Frames Consumed: {streamer.frames_consumed}")
    print(f"Total Frames Dropped:  {streamer.frames_dropped}")
    print(f"Final Status:          {streamer.status}")
    print("=" * 75)


if __name__ == "__main__":
    main()
