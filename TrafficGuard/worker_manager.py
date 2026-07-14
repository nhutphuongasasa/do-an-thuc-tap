"""
workers/worker_manager.py
============================
Central lifecycle manager for every worker pool in the pipeline.
Responsibilities:

    - Instantiate all queues.
    - Create the right NUMBER of workers per pool based on
      config.settings (auto-detected from CPU count).
    - start() everything in the correct order.
    - stop() everything gracefully (reverse order, so upstream
      producers stop before downstream consumers, minimizing lost
      in-flight data).
"""

import itertools
import logging
import threading
import time

from config import settings
from queues.packet_queue import PacketQueue
from queues.flow_queue import FlowQueue
from queues.feature_queue import FeatureQueue
from queues.ml_result_queue import MLResultQueue
from queues.alert_queue import AlertQueue

from workers.capture_worker import PacketCaptureWorker
from workers.flow_worker import FlowWorker
from workers.feature_worker import FeatureWorker
from workers.ml_worker import MLWorker

from suricata.eve_reader import SuricataEveReader
from correlation.risk_engine import RiskEngine
from alert.alert_manager import AlertManager

logger = logging.getLogger("nids.worker_manager")


class WorkerManager:
    def __init__(self):
        # ---------------- Queues ----------------
        self.packet_queue = PacketQueue()
        self.flow_queue = FlowQueue()
        self.feature_queue = FeatureQueue()
        self.ml_result_queue = MLResultQueue()
        self.alert_queue = AlertQueue()

        # ---------------- Capture ----------------
        self.capture_worker = PacketCaptureWorker(
            interface=settings.CAPTURE_INTERFACE,
            bpf_filter=settings.BPF_FILTER,
            packet_queue=self.packet_queue,
            put_timeout=settings.QUEUE_PUT_TIMEOUT,
        )

        # ---------------- Flow pool ----------------
        self.flow_workers = [
            FlowWorker(
                worker_id=i,
                packet_queue=self.packet_queue,
                flow_queue=self.flow_queue,
                idle_timeout=settings.FLOW_IDLE_TIMEOUT,
                hard_timeout=settings.FLOW_HARD_TIMEOUT,
                sweep_interval=settings.FLOW_SWEEP_INTERVAL,
                get_timeout=settings.QUEUE_GET_TIMEOUT,
                put_timeout=settings.QUEUE_PUT_TIMEOUT,
            )
            for i in range(settings.FLOW_WORKERS)
        ]

        # ---------------- Feature pool ----------------
        self.feature_workers = [
            FeatureWorker(
                worker_id=i,
                flow_queue=self.flow_queue,
                feature_queue=self.feature_queue,
                get_timeout=settings.QUEUE_GET_TIMEOUT,
                put_timeout=settings.QUEUE_PUT_TIMEOUT,
            )
            for i in range(settings.FEATURE_WORKERS)
        ]

        # ---------------- ML pool ----------------
        model_cycle = (
            itertools.cycle(settings.ML_MODEL_PATHS)
            if settings.ML_MODEL_PATHS else itertools.cycle([""])
        )
        self.ml_workers = [
            MLWorker(
                worker_id=i,
                model_path=next(model_cycle),
                feature_queue=self.feature_queue,
                ml_result_queue=self.ml_result_queue,
                get_timeout=settings.QUEUE_GET_TIMEOUT,
                put_timeout=settings.QUEUE_PUT_TIMEOUT,
                log_level=settings.LOG_LEVEL,
            )
            for i in range(settings.ML_WORKERS)
        ]

        # ---------------- Suricata side-channel ----------------
        self.suricata_reader = SuricataEveReader(settings.SURICATA_EVE_JSON)
        self._recent_rule_events: dict = {}
        self._rule_events_lock = threading.Lock()

        # ---------------- Correlation + Alerting ----------------
        self.risk_engine = RiskEngine(
            rule_weight=settings.RULE_SCORE_WEIGHT,
            ml_weight=settings.ML_SCORE_WEIGHT,
            low=settings.RISK_THRESHOLD_LOW,
            medium=settings.RISK_THRESHOLD_MEDIUM,
            high=settings.RISK_THRESHOLD_HIGH,
        )

        db_writer = None
        if settings.DB_WRITE_ENABLED and settings.DATABASE_URL:
            try:
                from database.postgres import PostgresWriter
                db_writer = PostgresWriter(settings.DATABASE_URL)
                db_writer.connect()
            except Exception:
                logger.exception("Could not connect to PostgreSQL, continuing without DB writes")
                db_writer = None

        self.alert_manager = AlertManager(
            json_path=settings.ALERT_JSON_PATH,
            log_path=settings.ALERT_LOG_PATH,
            threshold=settings.ALERT_THRESHOLD,
            db_writer=db_writer,
        )

        # background threads for suricata tailing + correlation loop
        self._stop_event = threading.Event()
        self._suricata_thread = threading.Thread(
            target=self._suricata_loop, name="SuricataReader", daemon=True
        )
        self._correlation_thread = threading.Thread(
            target=self._correlation_loop, name="CorrelationWorker", daemon=True
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        logger.info(
            "Starting worker pools: flow=%d feature=%d ml=%d (cpu_count=%d)",
            settings.FLOW_WORKERS, settings.FEATURE_WORKERS, settings.ML_WORKERS,
            settings.CPU_COUNT,
        )

        # Start consumers before producers so nothing is dropped
        # immediately at startup.
        for w in self.ml_workers:
            w.start()
        for w in self.feature_workers:
            w.start()
        for w in self.flow_workers:
            w.start()

        self._correlation_thread.start()
        self._suricata_thread.start()

        self.capture_worker.start()
        logger.info("All workers started")

    def stop(self):
        logger.info("Graceful shutdown initiated...")
        self._stop_event.set()

        # Stop producers first, then let queues drain downstream.
        self.capture_worker.stop()

        for w in self.flow_workers:
            w.stop()

        for w in self.feature_workers:
            w.stop()

        for w in self.ml_workers:
            w.stop()

        self.suricata_reader.stop()
        self._suricata_thread.join(timeout=5)
        self._correlation_thread.join(timeout=5)

        for q in (self.packet_queue, self.flow_queue, self.feature_queue,
                  self.ml_result_queue, self.alert_queue):
            q.stop()

        logger.info("Shutdown complete")

    # ------------------------------------------------------------------
    # Suricata rule events: cache by (src_ip,dst_ip,src_port,dst_port,proto)
    # so the correlation loop can look up a matching rule event for a
    # given ML result's flow context.
    # ------------------------------------------------------------------
    def _suricata_loop(self):
        try:
            for event in self.suricata_reader.tail():
                if self._stop_event.is_set():
                    break
                key = (
                    event.source_ip, event.destination_ip, event.source_port,
                    event.destination_port, event.protocol.upper(),
                )
                with self._rule_events_lock:
                    self._recent_rule_events[key] = event.to_dict()
        except Exception:
            logger.exception("Suricata reader loop crashed")

    def _lookup_rule_event(self, context: dict):
        key = (
            context.get("src_ip"), context.get("dst_ip"),
            context.get("src_port"), context.get("dst_port"),
            (context.get("protocol") or "").upper(),
        )
        with self._rule_events_lock:
            return self._recent_rule_events.pop(key, None)

    # ------------------------------------------------------------------
    # Correlation loop: consumes ML results, joins with any matching
    # Suricata rule event, scores via RiskEngine, hands to AlertManager.
    # ------------------------------------------------------------------
    def _correlation_loop(self):
        while not self._stop_event.is_set():
            ml_result = self.ml_result_queue.get(timeout=settings.QUEUE_GET_TIMEOUT)
            if ml_result is None:
                continue
            try:
                context = ml_result.get("_meta", {})
                rule_event = self._lookup_rule_event(context)
                risk_event = self.risk_engine.correlate(rule_event, ml_result, context)
                self.alert_manager.handle(risk_event)
            except Exception:
                logger.exception("Correlation loop failed to process an ML result")