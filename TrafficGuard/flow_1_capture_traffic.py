import threading
import time
import queue
import logging
from typing import Optional

from scapy.all import sniff, IP, TCP, UDP

from .models import PacketEvent

logger = logging.getLogger("nids.capture")


class PacketCaptureEngine:
    def __init__(self, out_queue: "queue.Queue[PacketEvent]",
                 interface: Optional[str] = None, bpf_filter: str = "ip"):
        self.out_queue = out_queue
        self.interface = interface
        self.bpf_filter = bpf_filter

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Luu chieu "forward" (ai la nguoi khoi tao) cho tung flow, de goi tra loi
        # duoc gan dung nhan "bwd".
        self._flow_origin = {}
        self._flow_origin_lock = threading.Lock()

        # So lieu thong ke de in ra man hinh dinh ky
        self.packet_count = 0
        self.dropped_count = 0

    # ---------------- helpers ----------------
    @staticmethod
    def _make_flow_key(src_ip, dst_ip, sport, dport, proto):
        """5-tuple flow key khong phan biet chieu: (A,portA) <-> (B,portB) luon ra 1 key."""
        if (src_ip, sport) <= (dst_ip, dport):
            return (src_ip, sport, dst_ip, dport, proto)
        return (dst_ip, dport, src_ip, sport, proto)

    def _direction(self, flow_key, src_ip, sport):
        with self._flow_origin_lock:
            origin = self._flow_origin.get(flow_key)
            if origin is None:
                self._flow_origin[flow_key] = (src_ip, sport)
                return "fwd"
            return "fwd" if origin == (src_ip, sport) else "bwd"

    @staticmethod
    def _extract_flags(tcp_layer):
        if tcp_layer is None:
            return {"FIN": 0, "SYN": 0, "RST": 0, "PSH": 0, "ACK": 0, "URG": 0}
        flags = int(tcp_layer.flags)
        return {
            "FIN": int(bool(flags & 0x01)),
            "SYN": int(bool(flags & 0x02)),
            "RST": int(bool(flags & 0x04)),
            "PSH": int(bool(flags & 0x08)),
            "ACK": int(bool(flags & 0x10)),
            "URG": int(bool(flags & 0x20)),
        }

    # ---------------- scapy callback ----------------
    def _on_packet(self, pkt):
        try:
            if IP not in pkt:
                return
            ip_layer = pkt[IP]

            if TCP in pkt:
                proto = "TCP"
                sport, dport = pkt[TCP].sport, pkt[TCP].dport
                tcp_layer = pkt[TCP]
            elif UDP in pkt:
                proto = "UDP"
                sport, dport = pkt[UDP].sport, pkt[UDP].dport
                tcp_layer = None
            else:
                proto = str(ip_layer.proto)
                sport, dport = 0, 0
                tcp_layer = None

            flow_key = self._make_flow_key(ip_layer.src, ip_layer.dst, sport, dport, proto)
            direction = self._direction(flow_key, ip_layer.src, sport)

            event = PacketEvent(
                ts=time.time(),
                flow_key=flow_key,
                direction=direction,
                length=len(pkt),
                flags=self._extract_flags(tcp_layer),
                protocol=proto,
            )

            try:
                self.out_queue.put_nowait(event)
                self.packet_count += 1
            except queue.Full:
                self.dropped_count += 1
                if self.dropped_count % 1000 == 1:
                    logger.warning("Queue1 dang day! Da drop %d packet tong cong.",
                                   self.dropped_count)

        except Exception:
            logger.exception("Loi khi xu ly 1 packet trong Flow1")

    def _run(self):
        logger.info("Flow1 (Packet Capture) bat dau. interface=%s filter='%s'",
                    self.interface or "<default>", self.bpf_filter)
        sniff(
            iface=self.interface,
            filter=self.bpf_filter,
            prn=self._on_packet,
            store=False,
            stop_filter=lambda _p: self._stop_event.is_set(),
        )
        logger.info("Flow1 (Packet Capture) da dung.")

    def start(self):
        self._thread = threading.Thread(target=self._run, name="Flow1-Capture", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)