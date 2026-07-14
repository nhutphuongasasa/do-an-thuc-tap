"""
queues/base_queue.py
=====================
Common bounded-queue implementation shared by packet/flow/feature/alert
queues.

Design goals (per spec):
- max size (backpressure instead of unbounded memory growth)
- put/get with timeout
- drop counter instead of per-event log spam
- periodic aggregated stats log line, e.g.:

      [QUEUE STATS] Feature dropped: 500 | Current size: 9000

This is intentionally a thin wrapper around `queue.Queue` today so it
can be swapped for a Redis Streams / Kafka backed implementation later
(see QUEUE_BACKEND in config.py) while keeping the same public API:
put(), get(), qsize(), stats().
"""

import logging
import multiprocessing
import queue
import threading
import time

logger = logging.getLogger("nids.queues")


class BaseBoundedQueue:
    """Thread-backed bounded queue (queue.Queue). Use this for stages
    that stay within a single process (e.g. Packet/Flow queues shared
    across threads)."""

    def __init__(self, name: str, maxsize: int, stats_interval: int = 10):
        self.name = name
        self._q = queue.Queue(maxsize=maxsize)
        self._dropped = 0
        self._lock = threading.Lock()
        self._stats_interval = stats_interval
        self._stop_stats = threading.Event()
        self._stats_thread = threading.Thread(
            target=self._stats_loop, name=f"{name}-stats", daemon=True
        )
        self._stats_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def put(self, item, timeout: float = 0.1) -> bool:
        """Try to enqueue an item. Returns False (and counts a drop)
        instead of raising/blocking forever when the queue is full."""
        try:
            self._q.put(item, timeout=timeout)
            return True
        except queue.Full:
            with self._lock:
                self._dropped += 1
            return False

    def get(self, timeout: float = 0.5):
        """Returns an item or None on timeout (never raises queue.Empty
        to callers, so worker loops stay simple)."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def qsize(self) -> int:
        return self._q.qsize()

    def stats(self) -> dict:
        with self._lock:
            return {"name": self.name, "dropped": self._dropped, "size": self._q.qsize()}

    def stop(self):
        self._stop_stats.set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _stats_loop(self):
        while not self._stop_stats.wait(self._stats_interval):
            s = self.stats()
            if s["dropped"] or s["size"]:
                logger.info(
                    "[QUEUE STATS] %s dropped: %d | Current size: %d",
                    self.name, s["dropped"], s["size"],
                )


class MPBoundedQueue:
    """multiprocessing.Queue-backed bounded queue with the same public
    API as BaseBoundedQueue. Required whenever a queue needs to cross
    a process boundary — e.g. Feature Queue -> ML Worker Pool
    (multiprocessing.Process workers), and ML results -> Correlation
    Worker back in the main process.

    Drop counting uses a multiprocessing.Value so it stays accurate
    across processes.
    """

    def __init__(self, name: str, maxsize: int, stats_interval: int = 10):
        self.name = name
        self._q = multiprocessing.Queue(maxsize=maxsize)
        self._dropped = multiprocessing.Value("i", 0)
        self._stats_interval = stats_interval
        self._stop_stats = threading.Event()
        self._stats_thread = threading.Thread(
            target=self._stats_loop, name=f"{name}-stats", daemon=True
        )
        self._stats_thread.start()

    def put(self, item, timeout: float = 0.1) -> bool:
        try:
            self._q.put(item, timeout=timeout)
            return True
        except queue.Full:
            with self._dropped.get_lock():
                self._dropped.value += 1
            return False

    def get(self, timeout: float = 0.5):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def qsize(self) -> int:
        try:
            return self._q.qsize()
        except NotImplementedError:
            return -1  # not supported on some platforms (e.g. macOS)

    def stats(self) -> dict:
        return {"name": self.name, "dropped": self._dropped.value, "size": self.qsize()}

    def stop(self):
        self._stop_stats.set()

    def _stats_loop(self):
        while not self._stop_stats.wait(self._stats_interval):
            s = self.stats()
            if s["dropped"] or (s["size"] and s["size"] > 0):
                logger.info(
                    "[QUEUE STATS] %s dropped: %d | Current size: %d",
                    self.name, s["dropped"], s["size"],
                )