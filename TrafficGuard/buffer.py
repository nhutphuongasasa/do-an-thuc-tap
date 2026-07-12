import threading
import time
from collections import deque, defaultdict


class FlowBuffer:
    def __init__(self, max_age_seconds: float):
        self.max_age = max_age_seconds
        self._lock = threading.Lock()
        self._flows = defaultdict(deque)   
        self._last_seen = {}              

    def add(self, pkt):
        with self._lock:
            dq = self._flows[pkt.flow_key]
            dq.append(pkt)
            self._last_seen[pkt.flow_key] = pkt.ts
            self._prune_flow(dq, pkt.ts)

    def _prune_flow(self, dq, now_ts):
        cutoff = now_ts - self.max_age
        while dq and dq[0].ts < cutoff:
            dq.popleft()

    def snapshot_window(self, flow_key, window_seconds, now_ts=None):
        now_ts = now_ts if now_ts is not None else time.time()
        cutoff = now_ts - window_seconds
        with self._lock:
            dq = self._flows.get(flow_key)
            if not dq:
                return []
            return [p for p in dq if p.ts >= cutoff]

    def active_flow_keys(self, idle_timeout):
        now = time.time()
        with self._lock:
            return [k for k, last in self._last_seen.items() if now - last <= idle_timeout]

    def cleanup_dead_flows(self, idle_timeout):
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