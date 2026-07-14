"""
capture/packet_parser.py
==========================
Turns a raw Scapy packet into a lightweight, plain-dict representation
that is cheap to pass through queues (avoids pickling full Scapy
objects across process/thread boundaries downstream).
"""

import time
from scapy.layers.inet import IP, TCP, UDP, ICMP


def parse_packet(pkt) -> dict | None:
    """Return a normalized dict for IP packets, or None if not IP
    (i.e. not something we track as a 5-tuple flow)."""
    if IP not in pkt:
        return None

    ip = pkt[IP]
    proto = "OTHER"
    src_port = 0
    dst_port = 0
    tcp_flags = None

    if TCP in pkt:
        proto = "TCP"
        src_port = int(pkt[TCP].sport)
        dst_port = int(pkt[TCP].dport)
        tcp_flags = str(pkt[TCP].flags)
    elif UDP in pkt:
        proto = "UDP"
        src_port = int(pkt[UDP].sport)
        dst_port = int(pkt[UDP].dport)
    elif ICMP in pkt:
        proto = "ICMP"

    return {
        "timestamp": time.time(),
        "src_ip": ip.src,
        "dst_ip": ip.dst,
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": proto,
        "length": int(len(pkt)),
        "tcp_flags": tcp_flags,
        "ttl": int(ip.ttl),
    }