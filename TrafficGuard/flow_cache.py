"""
flow/flow_cache.py
=====================
Thread-safe store of active flows, keyed by 5-tuple.

Each Flow Worker in the pool owns its OWN FlowCache instance (packets
are sharded across workers, see worker_manager sharding strategy), so
there's no cross-worker lock contention.
"""

import threading
import time

from flow.models import Flow, FlowKey


class FlowCache:
    def __init__(self, idle_timeout: int, hard_timeout: int):
        self.idle_timeout = idle_timeout
        self.hard_timeout = hard_timeout
        self._flows: dict[tuple, Flow] = {}
        self._lock = threading.Lock()

    def get_or_create(self, key: FlowKey) -> Flow:
        with self._lock:
            t = key.as_tuple()
            flow = self._flows.get(t)
            if flow is None:
                flow = Flow(key=key)
                self._flows[t] = flow
            return flow

    def remove(self, key: FlowKey):
        with self._lock:
            self._flows.pop(key.as_tuple(), None)

    def sweep_expired(self) -> list[Flow]:
        """Remove and return flows that have hit idle or hard timeout."""
        now = time.time()
        expired = []
        with self._lock:
            for t, flow in list(self._flows.items()):
                idle_for = now - flow.last_seen
                age = now - flow.start_time
                if idle_for >= self.idle_timeout or age >= self.hard_timeout:
                    expired.append(flow)
                    del self._flows[t]
        return expired

    def active_count(self) -> int:
        with self._lock:
            return len(self._flows)