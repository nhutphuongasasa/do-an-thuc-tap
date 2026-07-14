"""
capture/scapy_capture.py
==========================
Independent capture engine. Its ONLY job is:

    1. Sniff packets off the wire (via Scapy).
    2. Parse each packet into a plain dict.
    3. Push it onto the Packet Queue.

It never does flow tracking, feature extraction, or detection —
that separation is what lets the capture path stay fast and lets
downstream stages scale independently.
"""

import logging
import threading

from scapy.all import sniff

from capture.packet_parser import parse_packet

logger = logging.getLogger("nids.capture")


class ScapyCaptureEngine:
    def __init__(self, interface: str, bpf_filter: str, packet_queue, put_timeout: float = 0.1):
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.packet_queue = packet_queue
        self.put_timeout = put_timeout

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        logger.info("Starting capture on interface=%s filter='%s'", self.interface, self.bpf_filter)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="CaptureWorker", daemon=True)
        self._thread.start()

    def stop(self):
        logger.info("Stopping capture engine...")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _run(self):
        try:
            sniff(
                iface=self.interface,
                filter=self.bpf_filter,
                prn=self._on_packet,
                store=False,
                stop_filter=lambda _pkt: self._stop_event.is_set(),
            )
        except PermissionError:
            logger.error(
                "Permission denied opening interface %s. "
                "Capture requires elevated privileges (CAP_NET_RAW).",
                self.interface,
            )
        except Exception:
            logger.exception("Capture engine crashed")

    def _on_packet(self, pkt):
        try:
            parsed = parse_packet(pkt)
        except Exception:
            logger.debug("Failed to parse packet", exc_info=True)
            return

        if parsed is None:
            return

        self.packet_queue.put(parsed, timeout=self.put_timeout)