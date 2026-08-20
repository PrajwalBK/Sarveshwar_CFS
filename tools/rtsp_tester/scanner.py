"""
CP Plus & ONVIF IP Camera Auto-Discovery & Subnet Scanner
100% Offline | Finds Camera IP directly connected via Ethernet
"""

import socket
import struct
import threading
import subprocess
import sys
import re
from typing import List, Dict, Any, Optional


def get_local_interfaces() -> List[str]:
    """Get all local IPv4 addresses across network adapters."""
    ip_list = []
    try:
        # Try hostname lookup
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127."):
                ip_list.append(ip)
    except Exception:
        pass

    # Common default camera subnets to check even if laptop is on a different subnet
    default_subnets = ["192.168.1.100", "192.168.0.100", "192.168.100.100", "10.0.0.100"]
    for sub in default_subnets:
        if sub not in ip_list:
            ip_list.append(sub)

    return ip_list


def discover_onvif_ws(timeout_sec: float = 2.0) -> List[Dict[str, Any]]:
    """
    Discover cameras via ONVIF WS-Discovery (UDP 3702 to 239.255.255.250).
    Works on all ONVIF-compliant cameras including CP Plus.
    """
    found_cameras = []
    ws_msg = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<Envelope xmlns:dn="http://www.onvif.org/ver10/network/wsdl" '
        'xmlns="http://www.w3.org/2003/05/soap-envelope">'
        '<Header>'
        '<wsa:MessageID xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">'
        'uuid:c898b350-0a17-48f8-a1c2-3e2c347df277</wsa:MessageID>'
        '<wsa:To xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">'
        'urn:schemas-xmlsoap-org:ws:2005:04:discovery</wsa:To>'
        '<wsa:Action xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">'
        'http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</wsa:Action>'
        '</Header>'
        '<Body>'
        '<Probe xmlns="http://schemas.xmlsoap.org/ws/2005/04/discovery">'
        '<Types>dn:NetworkVideoTransmitter</Types>'
        '</Probe>'
        '</Body>'
        '</Envelope>'
    ).encode("utf-8")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(timeout_sec)

        # Send multicast discovery
        multicast_group = ("239.255.255.250", 3702)
        sock.sendto(ws_msg, multicast_group)

        # Send global broadcast fallback
        sock.sendto(ws_msg, ("255.255.255.255", 3702))

        while True:
            try:
                data, addr = sock.recvfrom(4096)
                ip = addr[0]
                text = data.decode("utf-8", errors="ignore")
                
                # Extract XAddrs or service URL
                url_match = re.search(r"http://([\d\.]+):?(\d+)?/", text)
                cam_ip = url_match.group(1) if url_match else ip

                if not any(c["ip"] == cam_ip for c in found_cameras):
                    found_cameras.append({
                        "ip": cam_ip,
                        "protocol": "ONVIF (WS-Discovery)",
                        "port": 554,
                    })
            except socket.timeout:
                break
            except Exception:
                break
        sock.close()
    except Exception:
        pass

    return found_cameras


def discover_cpplus_dahua(timeout_sec: float = 2.0) -> List[Dict[str, Any]]:
    """
    Discover cameras via CP Plus / Dahua Private Protocol (UDP 37810).
    """
    found_cameras = []
    # Dahua/CP Plus discovery broadcast packet
    dh_msg = b"\xa0\x00\x00\x60\x00\x00\x00\x00DHIP\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.settimeout(timeout_sec)

        sock.sendto(dh_msg, ("255.255.255.255", 37810))
        sock.sendto(dh_msg, ("255.255.255.255", 37777))

        while True:
            try:
                data, addr = sock.recvfrom(2048)
                ip = addr[0]
                if not any(c["ip"] == ip for c in found_cameras):
                    found_cameras.append({
                        "ip": ip,
                        "protocol": "CP Plus / Dahua Private",
                        "port": 554,
                    })
            except socket.timeout:
                break
            except Exception:
                break
        sock.close()
    except Exception:
        pass

    return found_cameras


def check_rtsp_port(ip: str, port: int = 554, timeout_sec: float = 0.25) -> bool:
    """Check if RTSP port 554 (or HTTP 80) is open on an IP."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout_sec)
        res = s.connect_ex((ip, port))
        s.close()
        return res == 0
    except Exception:
        return False


def scan_common_camera_ips() -> List[Dict[str, Any]]:
    """
    Fast parallel TCP sweep across the most common default IP camera addresses:
    - CP Plus defaults: 192.168.1.250, 192.168.0.250
    - Hikvision/Dahua defaults: 192.168.1.64, 192.168.1.108, 192.168.0.64
    - Gateway defaults: 192.168.1.1, 192.168.0.1, 192.168.100.1
    """
    common_ips = [
        "192.168.1.250",
        "192.168.0.250",
        "192.168.1.108",
        "192.168.0.108",
        "192.168.1.64",
        "192.168.0.64",
        "192.168.100.250",
        "192.168.1.10",
        "192.168.1.100",
        "192.168.0.100",
        "10.0.0.250",
    ]

    # Also scan active ARP table
    try:
        arp_out = subprocess.check_output("arp -a", shell=True).decode("utf-8", errors="ignore")
        for match in re.finditer(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", arp_out):
            candidate = match.group(1)
            if not candidate.endswith(".255") and not candidate.startswith("224.") and not candidate.startswith("239."):
                if candidate not in common_ips:
                    common_ips.append(candidate)
    except Exception:
        pass

    found = []
    threads = []
    lock = threading.Lock()

    def _worker(ip):
        # Test RTSP port 554 first, then HTTP port 80 / CP Plus 37777
        if check_rtsp_port(ip, 554, timeout_sec=0.3) or check_rtsp_port(ip, 37777, timeout_sec=0.3) or check_rtsp_port(ip, 80, timeout_sec=0.3):
            with lock:
                if not any(c["ip"] == ip for c in found):
                    found.append({
                        "ip": ip,
                        "protocol": "Port 554 / 37777 Open",
                        "port": 554,
                    })

    for ip in common_ips:
        t = threading.Thread(target=_worker, args=(ip,), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join(timeout=0.6)

    return found


def scan_network_for_cameras() -> List[Dict[str, Any]]:
    """
    Run all discovery methods in parallel and return unique list of found IP cameras.
    """
    results = []
    seen_ips = set()

    # 1. CP Plus / Dahua Private Protocol
    for cam in discover_cpplus_dahua(timeout_sec=1.5):
        if cam["ip"] not in seen_ips:
            seen_ips.add(cam["ip"])
            results.append(cam)

    # 2. ONVIF WS-Discovery
    for cam in discover_onvif_ws(timeout_sec=1.5):
        if cam["ip"] not in seen_ips:
            seen_ips.add(cam["ip"])
            results.append(cam)

    # 3. Fast Port & ARP Sweep
    for cam in scan_common_camera_ips():
        if cam["ip"] not in seen_ips:
            seen_ips.add(cam["ip"])
            results.append(cam)

    return results


if __name__ == "__main__":
    print("Scanning network for CP Plus / ONVIF IP cameras...")
    cams = scan_network_for_cameras()
    if cams:
        print(f"✓ Found {len(cams)} camera(s):")
        for c in cams:
            print(f"  - IP: {c['ip']} | Protocol: {c['protocol']} | RTSP Port: {c['port']}")
    else:
        print("✗ No cameras found. Check Ethernet cable and ensure laptop IP is set to 192.168.1.100.")
