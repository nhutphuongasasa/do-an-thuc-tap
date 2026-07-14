"""
zeek/zeek_parser.py
=====================
Tails Zeek's conn.log (TSV, JSON, or JSON-per-line depending on Zeek's
LogAscii writer config) and yields flow metadata records. This is used
as an optional enrichment source alongside the Scapy-derived flows —
Zeek is authoritative for protocol-aware connection state, byte/packet
counters, and service detection.

Supports Zeek's default JSON logging format (`json-lines` policy),
which is the simplest to integrate. If your Zeek writes TSV, convert
with `zeek-cut` or enable JSON logging via:

    @load policy/tuning/json-logs.zeek
"""

import json
import logging
import os
import time

logger = logging.getLogger("nids.zeek")


class ZeekConnLogTailer:
    def __init__(self, log_path: str, poll_interval: float = 1.0):
        self.log_path = log_path
        self.poll_interval = poll_interval
        self._stop = False
        self._fh = None
        self._pos = 0

    def stop(self):
        self._stop = True

    def _open(self):
        if not os.path.exists(self.log_path):
            return None
        fh = open(self.log_path, "r")
        fh.seek(0, os.SEEK_END)  # start at end, only tail new lines
        return fh

    def tail(self):
        """Generator yielding parsed conn.log records as dicts."""
        while not self._stop:
            if self._fh is None:
                self._fh = self._open()
                if self._fh is None:
                    time.sleep(self.poll_interval)
                    continue

            line = self._fh.readline()
            if not line:
                time.sleep(self.poll_interval)
                continue

            record = self._parse_line(line)
            if record:
                yield record

    @staticmethod
    def _parse_line(line: str) -> dict | None:
        line = line.strip()
        if not line:
            return None
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Skipping non-JSON zeek line (enable json-logs.zeek?)")
            return None

        return {
            "ts": raw.get("ts"),
            "uid": raw.get("uid"),
            "src_ip": raw.get("id.orig_h"),
            "src_port": raw.get("id.orig_p"),
            "dst_ip": raw.get("id.resp_h"),
            "dst_port": raw.get("id.resp_p"),
            "protocol": (raw.get("proto") or "").upper(),
            "service": raw.get("service"),
            "duration": raw.get("duration"),
            "orig_bytes": raw.get("orig_bytes"),
            "resp_bytes": raw.get("resp_bytes"),
            "conn_state": raw.get("conn_state"),
            "orig_pkts": raw.get("orig_pkts"),
            "resp_pkts": raw.get("resp_pkts"),
        }