"""
workers/flow_worker.py
=========================
One FlowWorker = one thread that:

    1. Pulls packets from the Packet Queue.
    2. Aggregates them into 5-tuple flows via FlowManager.
    3. Periodically sweeps timed-out flows and pushes them to the
       Flow Queue.

Multiple FlowWorkers run concurrently (FLOW_WORKERS, sized from CPU
count) — the Packet Queue is a shared thread-safe queue.Queue so
packets are naturally load-balanced across whichever worker grabs
them next. Threads (not processes) are used here since this stage is
I/O/queue-bound rather than CPU-bound.
"""

import logging
import threading
import time

from flow.flow_manager import FlowManager

logger = logging.getLogger("nids.worker.flow")


class FlowWorker:
    def __init__(
        self,
        worker_id: int,
        packet_queue,
        flow_queue,
        idle_timeout: int,
        hard_timeout: int,
        sweep_interval: int,
        get_timeout: float = 0.5,
        put_timeout: float = 0.1,
    ):
        self.worker_id = worker_id
        self.packet_queue = packet_queue
        self.flow_queue = flow_queue
        self.sweep_interval = sweep_interval
        self.get_timeout = get_timeout
        self.put_timeout = put_timeout

        self.manager = FlowManager(idle_timeout=idle_timeout, hard_timeout=hard_timeout)

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"FlowWorker-{worker_id}", daemon=True
        )

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5)

    def process_packet(self, pkt: dict):
        return self.manager.process_packet(pkt)

    def update_flow(self):
        # kept for API-compatibility with the requested class design;
        # actual per-packet updates happen inside process_packet()
        pass

    def flush_timeout_flow(self):
        completed = self.manager.flush_timeout_flows()
        for flow_dict in completed:
            self.flow_queue.put(flow_dict, timeout=self.put_timeout)

    def _run(self):
        logger.info("FlowWorker-%d started", self.worker_id)
        last_sweep = time.time()

        while not self._stop_event.is_set():
            pkt = self.packet_queue.get(timeout=self.get_timeout)
            if pkt is not None:
                self.process_packet(pkt)

            if time.time() - last_sweep >= self.sweep_interval:
                self.flush_timeout_flow()
                last_sweep = time.time()

        # final flush on shutdown
        self.flush_timeout_flow()
        logger.info("FlowWorker-%d stopped", self.worker_id)