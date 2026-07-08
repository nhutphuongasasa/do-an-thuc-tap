from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class PacketEvent:
    ts: float                 # thoi diem bat goi tin (epoch seconds)
    flow_key: Tuple           # (ip_a, port_a, ip_b, port_b, protocol) - da chuan hoa khong phan biet chieu
    direction: str            # "fwd" (ben khoi tao) hoac "bwd" (ben tra loi)
    length: int                # kich thuoc goi tin (byte)
    flags: Dict[str, int]      # {"FIN":0/1, "SYN":..., "RST":..., "PSH":..., "ACK":..., "URG":...}
    protocol: str              # "TCP" | "UDP" | so hieu protocol khac


@dataclass
class FeatureVector:
    flow_key: Tuple
    window_size: int           # 1, 3, 5 (giay) - window nao sinh ra vector nay
    ts: float                   # thoi diem tinh feature
    values: Dict[str, float]    # ten feature (giong luc train) -> gia tri


@dataclass
class Prediction:
    flow_key: Tuple
    window_size: int
    ts: float
    label: str
    confidence: float