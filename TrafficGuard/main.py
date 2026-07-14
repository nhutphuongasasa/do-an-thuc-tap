"""
main.py
========
Entrypoint for darktrace-nids.

    python main.py

Data flow (see README.md for full details):

    NIC -> PacketCaptureWorker -> Packet Queue
        -> Flow Worker Pool -> Flow Queue
        -> Feature Worker Pool -> Feature Queue
        -> ML Worker Pool (multiprocessing) -> ML Result Queue
        -> Correlation Worker (joins with Suricata eve.json events)
        -> Alert Manager -> alerts.json / alerts.log / PostgreSQL

Suricata's eve.json is tailed independently and joined into the
correlation step by 5-tuple.
"""

import logging
import os
import signal
import sys
import time

from config import settings
from workers.worker_manager import WorkerManager


def setup_logging():
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    log_file = os.path.join(settings.LOG_DIR, "nids.log")

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )


def main():
    setup_logging()
    logger = logging.getLogger("nids.main")

    logger.info("=" * 60)
    logger.info("Starting darktrace-nids")
    logger.info("CPU count detected: %d", settings.CPU_COUNT)
    logger.info(
        "Worker pools -> flow=%d feature=%d ml=%d",
        settings.FLOW_WORKERS, settings.FEATURE_WORKERS, settings.ML_WORKERS,
    )
    logger.info("=" * 60)

    manager = WorkerManager()

    shutdown_requested = {"flag": False}

    def _handle_signal(signum, _frame):
        if shutdown_requested["flag"]:
            logger.warning("Force exiting")
            sys.exit(1)
        logger.info("Received signal %s, shutting down gracefully...", signum)
        shutdown_requested["flag"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    manager.start()

    try:
        while not shutdown_requested["flag"]:
            time.sleep(1)
    finally:
        manager.stop()
        logger.info("darktrace-nids stopped")


if __name__ == "__main__":
    main()