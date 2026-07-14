"""
flow/flow_manager.py
======================
Business logic for turning packets into 5-tuple flows. Used inside
each FlowWorker instance (one FlowManager per worker/thread).
"""

import logging

from flow.flow_cache import FlowCache
from flow.models import FlowKey

logger = logging.getLogger("nids.flow")


class FlowManager:
    def __init__(self, idle_timeout: int, hard_timeout: int):
        self.cache = FlowCache(idle_timeout=idle_timeout, hard_timeout=hard_timeout)

    def process_packet(self, pkt: dict):
        key = FlowKey(
            src_ip=pkt["src_ip"],
            dst_ip=pkt["dst_ip"],
            src_port=pkt["src_port"],
            dst_port=pkt["dst_port"],
            protocol=pkt["protocol"],
        )
        flow = self.cache.get_or_create(key)
        flow.add_packet(pkt)
        return flow

    def flush_timeout_flows(self) -> list[dict]:
        """Return completed flows (idle/hard timeout) as plain dicts,
        ready to be pushed onto the Flow Queue."""
        expired = self.cache.sweep_expired()
        return [f.to_dict() for f in expired]

    def active_flow_count(self) -> int:
        return self.cache.active_count()