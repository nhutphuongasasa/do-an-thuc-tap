"""
flow_aggregator.py — Gom gói tin thành flow theo 5-tuple.

Hỗ trợ:
- Kết thúc flow khi FIN/RST hoặc timeout
- Time-window aggregation (config time_window_sec)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from pipeline.feature_extractor import FlowRecord, extract_flow_features


class FlowAggregator:
    """Gom packet → flow theo 5-tuple (src_ip, dst_ip, src_port, dst_port, proto)."""

    def __init__(
        self,
        flow_timeout: float = 30.0,
        time_window_sec: float | None = None,
        on_flow_ready: Callable[[FlowRecord, np.ndarray], None] | None = None,
    ):
        self.flow_timeout = flow_timeout
        self.time_window_sec = time_window_sec
        self.on_flow_ready = on_flow_ready
        self.active_flows: dict[tuple, FlowRecord] = {}

    def add_packet(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: str,
        timestamp: float,
        header_len: int,
        payload_len: int,
        flags: str,
        win_bytes: int,
    ) -> None:
        forward_key = (src_ip, dst_ip, src_port, dst_port, protocol)
        backward_key = (dst_ip, src_ip, dst_port, src_port, protocol)

        if forward_key in self.active_flows:
            flow_key, is_forward = forward_key, True
        elif backward_key in self.active_flows:
            flow_key, is_forward = backward_key, False
        else:
            flow_key = forward_key
            is_forward = True
            self.active_flows[flow_key] = FlowRecord(
                src_ip, dst_ip, src_port, dst_port, protocol
            )

        flow = self.active_flows[flow_key]
        flow.add_packet(timestamp, is_forward, header_len, payload_len, flags, win_bytes)

        self._flush_expired(timestamp)

    def _flush_expired(self, current_time: float) -> None:
        finished: list[tuple] = []
        for key, flow in self.active_flows.items():
            timed_out = flow.last_time is not None and (
                current_time - flow.last_time > self.flow_timeout
            )
            window_expired = (
                self.time_window_sec is not None
                and flow.start_time is not None
                and current_time - flow.start_time >= self.time_window_sec
            )
            if flow.is_finished or timed_out or window_expired:
                finished.append(key)

        for key in finished:
            self._emit_flow(self.active_flows.pop(key))

    def flush_all(self) -> None:
        """Xử lý toàn bộ flow còn lại (cuối file pcap)."""
        for flow in list(self.active_flows.values()):
            self._emit_flow(flow)
        self.active_flows.clear()

    def _emit_flow(self, flow: FlowRecord) -> None:
        features = extract_flow_features(flow)
        if features is not None and self.on_flow_ready:
            self.on_flow_ready(flow, features)
