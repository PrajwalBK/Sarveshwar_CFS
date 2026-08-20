"""
Advanced Hardware Link, Packet Listener & Broad Subnet Diagnostic Tool
Catches camera IP regardless of what subnet it is on (including APIPA 169.254.x.x, 192.168.29.x, etc.)
"""

import subprocess
import socket
import re
import sys
import threading
import time


def check_physical_ethernet_link():
    """Check physical link status on Windows using PowerShell."""
    print("\n[Check 1] Inspecting Physical Ethernet Link Status...")
    try:
        cmd = 'powershell "Get-NetAdapter | Select-Object Name, InterfaceDescription, Status, LinkSpeed"'
        out = subprocess.check_output(cmd, shell=True).decode("utf-8", errors="ignore")
        print(out.strip())
        
        if "Up" in out:
            print("  ✓ Ethernet Physical Link is UP (Connected).")
        elif "Disconnected" in out or "Disabled" in out:
            print("  ⚠️  Ethernet status shows Disconnected. Check the cable between PoE Injector and Laptop.")
    except Exception as e:
        print(f"  Error checking link: {e}")


def listen_for_camera_broadcasts(timeout_sec: float = 6.0):
    """Listen for UDP/DHCP/ARP broadcasts from the camera to capture its IP."""
    print(f"\n[Check 2] Listening for Camera Network Broadcasts ({timeout_sec}s)...")
    detected_ips = []
    
    ports_to_listen = [3702, 37810, 37777, 67, 68, 1900]
    sockets = []
    
    for port in ports_to_listen:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.settimeout(0.5)
            sockets.append((port, s))
        except Exception:
            pass

    t_end = time.monotonic() + timeout_sec
    while time.monotonic() < t_end:
        for port, s in sockets:
            try:
                data, addr = s.recvfrom(2048)
                ip = addr[0]
                if not ip.startswith("127.") and ip not in detected_ips:
                    detected_ips.append(ip)
                    print(f"  🎯 CAUGHT BROADCAST from IP: {ip} on Port {port}!")
            except socket.timeout:
                pass
            except Exception:
                pass
        time.sleep(0.05)

    for _, s in sockets:
        try:
            s.close()
        except Exception:
            pass

    return detected_ips


def scan_subnet(subnet_prefix: str) -> list:
    """Fast parallel probe of a /24 subnet."""
    found = []
    threads = []
    lock = threading.Lock()

    def _probe(ip):
        for port in (554, 37777, 80, 8000):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.18)
                res = s.connect_ex((ip, port))
                s.close()
                if res == 0:
                    with lock:
                        found.append((ip, port))
                    break
            except Exception:
                pass

    for i in range(1, 255):
        ip = f"{subnet_prefix}.{i}"
        t = threading.Thread(target=_probe, args=(ip,), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=0.3)

    return found


def main():
    print("=" * 75)
    print("🔍 Comprehensive CP Plus Camera Diagnostic & Sniffer")
    print("=" * 75)

    # 1. Physical Link Check
    check_physical_ethernet_link()

    # 2. Sniff Broadcast Packets
    broadcast_ips = listen_for_camera_broadcasts(timeout_sec=5.0)

    # 3. Sweep alternate subnets (192.168.29.x, 192.168.100.x, 169.254.1.x)
    print("\n[Check 3] Scanning Alternate Subnets (192.168.29.x, 192.168.100.x)...")
    found_alt = scan_subnet("192.168.29") + scan_subnet("192.168.100") + scan_subnet("169.254.1")

    # 4. Check ARP Table
    print("\n[Check 4] Checking System ARP Table (arp -a)...")
    try:
        arp_out = subprocess.check_output("arp -a", shell=True).decode("utf-8", errors="ignore")
        for line in arp_out.splitlines():
            match = re.search(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\s+([0-9a-fA-F\-]{17})", line)
            if match:
                ip, mac = match.groups()
                if not ip.endswith(".255") and not ip.startswith("224.") and not ip.startswith("239."):
                    print(f"  ✓ Device in ARP Table: IP = {ip:<16} MAC = {mac}")
    except Exception as e:
        print(f"  ARP check error: {e}")

    print("\n" + "=" * 75)
    if broadcast_ips or found_alt:
        print("🎯 DISCOVERED CAMERA IP(S):")
        for ip in broadcast_ips:
            print(f"   ▶ {ip}  (Captured from live camera broadcast)")
        for ip, port in found_alt:
            print(f"   ▶ {ip}  (Open port: {port})")
    else:
        print("📋 DIAGNOSTIC SUMMARY:")
        print("1. If 'Status' above showed 'Disconnected', Windows does not detect the physical cable link.")
        print("2. Check the PoE Injector:")
        print("   - Cable 1: Camera <---> PoE Injector (Port marked 'PoE' / 'P+D')")
        print("   - Cable 2: Laptop Ethernet <---> PoE Injector (Port marked 'LAN' / 'Data In')")
        print("3. If the camera has a reset button (usually near the SD card slot), holding it for 10s resets it to factory IP: 192.168.1.250.")
    print("=" * 75)


if __name__ == "__main__":
    main()
