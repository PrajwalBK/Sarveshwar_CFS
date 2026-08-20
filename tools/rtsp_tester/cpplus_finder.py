"""
Multi-Vendor IP Camera Packet Broadcast & ConfigTool Finder
Sends CP Plus / Dahua, Hikvision, ONVIF, and UPnP discovery probes across all network sockets.
"""

import socket
import time
import json
import re


def find_camera():
    print("=" * 75)
    print("📡 Sending Multi-Vendor Discovery Broadcasts to Ethernet...")
    print("=" * 75)

    found = []

    # 1. CP Plus / Dahua JSON Discovery (Used by CP Plus IP Config Tool)
    cpplus_json_payload = json.dumps({
        "method": "client.search",
        "params": {"mac": ""}
    }).encode("utf-8")

    # 2. CP Plus / Dahua Binary Discovery
    cpplus_bin_payload = b"\xa0\x00\x00\x60\x00\x00\x00\x00DHIP\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"

    # 3. ONVIF WS-Discovery Probe
    onvif_probe = (
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

    # 4. UPnP M-SEARCH Probe
    upnp_probe = (
        "M-SEARCH * HTTP/1.1\r\n"
        "HOST: 239.255.255.250:1900\r\n"
        'MAN: "ssdp:discover"\r\n'
        "MX: 2\r\n"
        "ST: ssdp:all\r\n\r\n"
    ).encode("utf-8")

    targets = [
        # (payload, host, port)
        (cpplus_json_payload, "255.255.255.255", 37810),
        (cpplus_json_payload, "239.255.255.250", 37810),
        (cpplus_bin_payload, "255.255.255.255", 37810),
        (cpplus_bin_payload, "255.255.255.255", 37777),
        (onvif_probe, "239.255.255.250", 3702),
        (onvif_probe, "255.255.255.255", 3702),
        (upnp_probe, "239.255.255.250", 1900),
    ]

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(1.5)

    for payload, host, port in targets:
        try:
            sock.sendto(payload, (host, port))
        except Exception:
            pass

    t_end = time.monotonic() + 4.0
    while time.monotonic() < t_end:
        try:
            data, addr = sock.recvfrom(4096)
            ip = addr[0]
            if ip.startswith("127."):
                continue

            raw_str = data.decode("utf-8", errors="ignore")
            
            # Extract details
            info = {"ip": ip, "raw": raw_str[:150]}
            
            # Check for JSON response (CP Plus ConfigTool protocol)
            try:
                js = json.loads(raw_str)
                if "params" in js:
                    p = js["params"]
                    info["ip"] = p.get("IPv4Address", {}).get("IPAddress", ip)
                    info["mac"] = p.get("MAC", "")
                    info["model"] = p.get("DeviceType", "")
            except Exception:
                pass

            if not any(f["ip"] == info["ip"] for f in found):
                found.append(info)
                print(f"\n🎯 CAMERA FOUND!")
                print(f"   ▶ IP Address: {info['ip']}")
                if "mac" in info:
                    print(f"   ▶ MAC:        {info['mac']}")
                if "model" in info:
                    print(f"   ▶ Model:      {info['model']}")
        except socket.timeout:
            pass
        except Exception:
            pass

    sock.close()

    if not found:
        print("\n⚠️  No camera responded to broadcast probes.")
        print("\nCommon Troubleshooting:")
        print("1. Did you hear the camera lens click on power-on?")
        print("2. If the camera has a physical RESET button (under the waterproof cap or near micro-SD slot), press and hold it for 10 seconds to restore factory default IP: 192.168.1.250.")
        print("3. Try setting your laptop Ethernet IP to 192.168.0.100 (in case the camera is on 192.168.0.250).")
    print("=" * 75)


if __name__ == "__main__":
    find_camera()
