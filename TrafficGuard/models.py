"""
flow/models.py
================
Data structures for flow tracking.
"""

from dataclasses import dataclass, field
import time


@dataclass(frozen=True)
class FlowKey:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str

    def as_tuple(self):
        return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol)


@dataclass
class Flow:
    key: FlowKey
    start_time: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    packet_count: int = 0
    byte_count: int = 0

    # timestamps of each packet, used later for inter-arrival-time (iat)
    # feature computation. Kept lightweight (list of floats).
    packet_timestamps: list = field(default_factory=list)
    packet_sizes: list = field(default_factory=list)
    tcp_flags_seen: list = field(default_factory=list)

    def add_packet(self, pkt: dict):
        self.packet_count += 1
        self.byte_count += pkt.get("length", 0)
        self.last_seen = pkt.get("timestamp", time.time())
        self.packet_timestamps.append(self.last_seen)
        self.packet_sizes.append(pkt.get("length", 0))
        if pkt.get("tcp_flags"):
            self.tcp_flags_seen.append(pkt["tcp_flags"])

    def duration(self) -> float:
        return max(0.0, self.last_seen - self.start_time)

    def to_dict(self) -> dict:
        return {
            "src_ip": self.key.src_ip,
            "dst_ip": self.key.dst_ip,
            "src_port": self.key.src_port,
            "dst_port": self.key.dst_port,
            "protocol": self.key.protocol,
            "start_time": self.start_time,
            "last_seen": self.last_seen,
            "duration": self.duration(),
            "packet_count": self.packet_count,
            "byte_count": self.byte_count,
            "packet_timestamps": self.packet_timestamps,
            "packet_sizes": self.packet_sizes,
            "tcp_flags_seen": self.tcp_flags_seen,
        }