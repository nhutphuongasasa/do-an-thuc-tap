"""
suricata/eve_reader.py
========================
Tails Suricata's eve.json output and yields RuleEvent objects for
`event_type == "alert"` records. This is the signature-detection path
that runs in parallel with the ML flow-analysis path (see architecture
diagram: Suricata Engine sits alongside the Flow Worker Pool).
"""

import json
import logging
import os
import time

from suricata.rule_event import RuleEvent

logger = logging.getLogger("nids.suricata")


class SuricataEveReader:
    def __init__(self, eve_path: str, poll_interval: float = 0.5):
        self.eve_path = eve_path
        self.poll_interval = poll_interval
        self._stop = False
        self._fh = None

    def stop(self):
        self._stop = True

    def _open(self):
        if not os.path.exists(self.eve_path):
            return None
        fh = open(self.eve_path, "r")
        fh.seek(0, os.SEEK_END)
        return fh

    def tail(self):
        """Generator yielding RuleEvent objects for alert events."""
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

            event = self._parse_line(line)
            if event:
                yield event

    @staticmethod
    def _parse_line(line: str) -> RuleEvent | None:
        line = line.strip()
        if not line:
            return None
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            return None

        if raw.get("event_type") != "alert":
            return None

        alert = raw.get("alert", {})
        try:
            return RuleEvent(
                signature=alert.get("signature", "unknown"),
                severity=int(alert.get("severity", 3)),
                category=alert.get("category", "unknown"),
                source_ip=raw.get("src_ip", ""),
                destination_ip=raw.get("dest_ip", ""),
                source_port=int(raw.get("src_port", 0) or 0),
                destination_port=int(raw.get("dest_port", 0) or 0),
                protocol=raw.get("proto", ""),
                timestamp=raw.get("timestamp", ""),
            )
        except Exception:
            logger.debug("Failed to parse suricata alert line", exc_info=True)
            return None