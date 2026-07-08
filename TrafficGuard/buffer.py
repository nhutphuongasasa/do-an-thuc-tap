import threading
import time
from collections import deque, defaultdict


class FlowBuffer:
    def __init__(self, max_age_seconds: float):
        self.max_age = max_age_seconds
        self._lock = threading.Lock()
        self._flows = defaultdict(deque)   # flow_key -> deque[PacketEvent]
        self._last_seen = {}               # flow_key -> ts goi tin cuoi cung

    def add(self, pkt):
        """Them 1 PacketEvent vao buffer (goi tu Aggregator thread)."""
        with self._lock:
            dq = self._flows[pkt.flow_key]
            dq.append(pkt)
            self._last_seen[pkt.flow_key] = pkt.ts
            self._prune_flow(dq, pkt.ts)

    def _prune_flow(self, dq, now_ts):
        """Xoa cac goi tin qua cu (qua max_age) khoi 1 flow - goi khi da giu lock."""
        cutoff = now_ts - self.max_age
        while dq and dq[0].ts < cutoff:
            dq.popleft()

    def snapshot_window(self, flow_key, window_seconds, now_ts=None):
        """Tra ve list PacketEvent cua flow_key trong [now - window_seconds, now]."""
        now_ts = now_ts if now_ts is not None else time.time()
        cutoff = now_ts - window_seconds
        with self._lock:
            dq = self._flows.get(flow_key)
            if not dq:
                return []
            # copy ra ngoai truoc khi tra ve de tranh giu lock lau
            return [p for p in dq if p.ts >= cutoff]

    def active_flow_keys(self, idle_timeout):
        """Danh sach flow con 'song' (co goi tin trong vong idle_timeout giay gan day)."""
        now = time.time()
        with self._lock:
            return [k for k, last in self._last_seen.items() if now - last <= idle_timeout]

    def cleanup_dead_flows(self, idle_timeout):
        """Don dep cac flow da timeout han de khong bi leak bo nho theo thoi gian."""
        now = time.time()
        with self._lock:
            dead = [k for k, last in self._last_seen.items() if now - last > idle_timeout]
            for k in dead:
                self._flows.pop(k, None)
                self._last_seen.pop(k, None)
        return len(dead)

    def stats(self):
        with self._lock:
            return {"active_flows": len(self._flows)}