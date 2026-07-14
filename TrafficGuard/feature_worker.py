"""
workers/feature_worker.py
============================
One FeatureWorker = one thread that pulls completed flows off the
Flow Queue, extracts a FeatureVector, and pushes it onto the Feature
Queue for ML inference.
"""

import logging
import threading

from features.feature_extractor import FeatureExtractor

logger = logging.getLogger("nids.worker.feature")


class FeatureWorker:
    def __init__(
        self,
        worker_id: int,
        flow_queue,
        feature_queue,
        get_timeout: float = 0.5,
        put_timeout: float = 0.1,
    ):
        self.worker_id = worker_id
        self.flow_queue = flow_queue
        self.feature_queue = feature_queue
        self.get_timeout = get_timeout
        self.put_timeout = put_timeout

        self.extractor = FeatureExtractor()

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"FeatureWorker-{worker_id}", daemon=True
        )

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5)

    def extract(self, flow: dict) -> dict:
        return self.extractor.extract(flow)

    def _run(self):
        logger.info("FeatureWorker-%d started", self.worker_id)
        while not self._stop_event.is_set():
            flow = self.flow_queue.get(timeout=self.get_timeout)
            if flow is None:
                continue
            try:
                feature_vector = self.extract(flow)
                self.feature_queue.put(feature_vector, timeout=self.put_timeout)
            except Exception:
                logger.exception("FeatureWorker-%d failed to process flow", self.worker_id)
        logger.info("FeatureWorker-%d stopped", self.worker_id)