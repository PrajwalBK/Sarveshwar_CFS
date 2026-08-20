#!/usr/bin/env python3
"""
Auto-discover GPS device across serial/UART ports and print live data.

Usage examples:
  python3 tools/test_gps_auto_discovery.py
  python3 tools/test_gps_auto_discovery.py --timeout 40
  python3 tools/test_gps_auto_discovery.py --bauds 4800,9600,38400,115200
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple

import serial

# Add project root for imports when run from tools/
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.gps_serial_utils import discover_gps_serial_ports

try:
    import pynmea2
except ImportError:
    pynmea2 = None


def _is_nmea_line(line: str) -> bool:
    return line.startswith("$GP") or line.startswith("$GN") or line.startswith("$BD")


def _read_one_fix(port: str, baud: int, timeout_s: int) -> Optional[Tuple[str, Optional[object]]]:
    """
    Read from (port, baud) until a likely NMEA sentence is seen.
    Returns (line, parsed_message_or_none) on success, else None.
    """
    deadline = time.time() + timeout_s
    try:
        with serial.Serial(port, baudrate=baud, timeout=1) as ser:
            while time.time() < deadline:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not _is_nmea_line(line):
                    continue

                parsed = None
                if pynmea2 is not None:
                    try:
                        parsed = pynmea2.parse(line)
                    except Exception:
                        parsed = None

                return line, parsed
    except (serial.SerialException, PermissionError, OSError):
        return None
    return None


def _iter_bauds(csv_values: str) -> Iterable[int]:
    for token in csv_values.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            yield int(token)
        except ValueError:
            continue


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover GPS serial device and print received GPS data."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=8,
        help="Probe timeout (seconds) per port/baud pair (default: 8)",
    )
    parser.add_argument(
        "--bauds",
        default="4800,9600,38400,115200",
        help="Comma-separated baud rates to test (default: 4800,9600,38400,115200)",
    )
    parser.add_argument(
        "--stream-seconds",
        type=int,
        default=20,
        help="After detection, stream live GPS lines for this many seconds (default: 20)",
    )
    args = parser.parse_args()

    bauds = list(_iter_bauds(args.bauds))
    if not bauds:
        print("ERROR: No valid baud rates provided.")
        return 2

    ports = discover_gps_serial_ports()
    print("=" * 70)
    print("GPS Auto Discovery")
    print("=" * 70)
    print(f"Discovered serial/UART ports: {len(ports)}")
    for p in ports:
        print(f"  - {p}")
    print(f"Baud rates to test: {bauds}")
    print()

    if not ports:
        print("No candidate serial ports found.")
        print("Tip: check permissions (dialout), wiring, and /dev/tty* availability.")
        return 1

    selected_port = None
    selected_baud = None

    for port in ports:
        for baud in bauds:
            print(f"[probe] {port} @ {baud} ...", end="", flush=True)
            result = _read_one_fix(port, baud, args.timeout)
            if result is None:
                print(" no NMEA")
                continue

            line, parsed = result
            selected_port = port
            selected_baud = baud
            print(" OK")
            print(f"  first NMEA: {line}")
            if parsed is not None:
                lat = getattr(parsed, "latitude", None)
                lon = getattr(parsed, "longitude", None)
                if lat and lon:
                    print(f"  parsed fix: lat={lat:.6f}, lon={lon:.6f}")
            break
        if selected_port:
            break

    if not selected_port:
        print("\nNo GPS NMEA data detected on discovered ports/bauds.")
        print("Try a longer timeout or include more bauds, e.g. --bauds 4800,9600,38400,57600,115200")
        return 1

    print("\n" + "=" * 70)
    print(f"Streaming GPS from {selected_port} @ {selected_baud} for {args.stream_seconds}s")
    print("=" * 70)

    end_stream = time.time() + args.stream_seconds
    try:
        with serial.Serial(selected_port, baudrate=selected_baud, timeout=1) as ser:
            while time.time() < end_stream:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="ignore").strip()
                if not _is_nmea_line(line):
                    continue
                stamp = datetime.utcnow().strftime("%H:%M:%S")
                if pynmea2 is not None:
                    try:
                        msg = pynmea2.parse(line)
                        lat = getattr(msg, "latitude", None)
                        lon = getattr(msg, "longitude", None)
                        if lat and lon:
                            print(f"[{stamp}] {line} | lat={lat:.6f}, lon={lon:.6f}")
                            continue
                    except Exception:
                        pass
                print(f"[{stamp}] {line}")
    except (serial.SerialException, PermissionError, OSError) as e:
        print(f"Failed to stream GPS on selected port: {e}")
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
