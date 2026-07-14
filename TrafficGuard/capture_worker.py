"""
workers/capture_worker.py
============================
Thin worker wrapper around capture.scapy_capture.ScapyCaptureEngine,
matching the class design requested in the spec:

    PacketCaptureWorker:
        start()
        stop()
        callback(packet)
"""

import logging

from capture.scapy_capture import ScapyCaptureEngine

logger = logging.getLogger("nids.worker.capture")


class PacketCaptureWorker:
    def __init__(self, interface: str, bpf_filter: str, packet_queue, put_timeout: float = 0.1):
        self.packet_queue = packet_queue
        self._engine = ScapyCaptureEngine(
            interface=interface,
            bpf_filter=bpf_filter,
            packet_queue=packet_queue,
            put_timeout=put_timeout,
        )

    def start(self):
        logger.info("PacketCaptureWorker starting")
        self._engine.start()

    def stop(self):
        logger.info("PacketCaptureWorker stopping")
        self._engine.stop()

    def callback(self, packet: dict):
        """Exposed for testing / manual injection without a live NIC —
        pushes an already-parsed packet dict straight into the queue."""
        self.packet_queue.put(packet, timeout=0.1)

    def is_running(self) -> bool:
        return self._engine.is_running()