import queue
import threading
import time
import signal
import logging
import sys

from nids import config
from nids.capture import PacketCaptureEngine
from nids.flow_buffer import FlowBuffer
from nids.feature_extractor import WindowFeatureExtractor
from nids.inference_engine import InferenceEngine
from nids.alerting import AlertManager


def setup_logging():
    import os
    os.makedirs(config.LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(config.SYSTEM_LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    setup_logging()
    logger = logging.getLogger("nids.main")
    logger.info("=" * 70)
    logger.info("KHOI DONG NIDS REALTIME PIPELINE (4 FLOW)")
    logger.info("Windows thoi gian: %s | Model dir: %s", config.WINDOW_SIZES, config.MODEL_DIR)
    logger.info("=" * 70)

    # ---- Queue giua cac Flow ----
    queue1 = queue.Queue(maxsize=config.QUEUE1_MAXSIZE)          # raw packet (Flow1 -> Aggregator)
    queue2 = queue.Queue(maxsize=config.QUEUE2_MAXSIZE)          # feature vector (Flow2 -> Flow3)
    alert_queue = queue.Queue(maxsize=config.ALERT_QUEUE_MAXSIZE)  # prediction (Flow3 -> Flow4)

    flow_buffer = FlowBuffer(max_age_seconds=config.MAX_BUFFER_SECONDS)

    # ---- FLOW 1: Packet Capture ----
    capture = PacketCaptureEngine(
        out_queue=queue1,
        interface=config.INTERFACE,
        bpf_filter=config.BPF_FILTER,
    )

    # ---- Aggregator: chuyen tu Queue1 vao FlowBuffer (dung chung cho moi Flow2 worker) ----
    stop_flag = threading.Event()

    def aggregator_loop():
        while not stop_flag.is_set():
            try:
                pkt = queue1.get(timeout=0.5)
            except queue.Empty:
                continue
            flow_buffer.add(pkt)

    aggregator_thread = threading.Thread(target=aggregator_loop, name="Aggregator", daemon=True)

    # ---- FLOW 2: 1 thread rieng cho moi window (1s, 3s, 5s...) ----
    window_workers = [
        WindowFeatureExtractor(
            flow_buffer=flow_buffer,
            out_queue=queue2,
            window_seconds=w,
            extract_interval=config.EXTRACT_INTERVAL,
            idle_threshold=config.ACTIVE_IDLE_THRESHOLD,
            flow_timeout=config.FLOW_TIMEOUT,
        )
        for w in config.WINDOW_SIZES
    ]

    # ---- FLOW 3: Inference Engine ----
    inference = InferenceEngine(
        in_queue=queue2,
        alert_queue=alert_queue,
        model_path=config.MODEL_PATH,
        scaler_path=config.SCALER_PATH,
        label_encoder_path=config.LABEL_ENCODER_PATH,
        feature_list_path=config.FEATURE_LIST_PATH,
        benign_label=config.BENIGN_LABEL,
        vote_mode=config.VOTE_MODE,
        window_weights=config.WINDOW_WEIGHTS,
        min_confidence=config.ALERT_MIN_CONFIDENCE,
    )

    # ---- FLOW 4: Alert Manager ----
    alert_manager = AlertManager(
        in_queue=alert_queue,
        log_dir=config.LOG_DIR,
        alert_csv_path=config.ALERT_LOG_FILE,
        benign_label=config.BENIGN_LABEL,
        cooldown_seconds=config.ALERT_COOLDOWN_SECONDS,
        telegram_token=config.TELEGRAM_BOT_TOKEN,
        telegram_chat_id=config.TELEGRAM_CHAT_ID,
    )

    # ---- Khoi dong theo dung thu tu: consumer truoc, producer (capture) sau cung ----
    aggregator_thread.start()
    for w in window_workers:
        w.start()
    inference.start()
    alert_manager.start()
    capture.start()

    logger.info("Pipeline da chay xong. Nhan Ctrl+C de dung an toan.")

    def shutdown(*_args):
        logger.info("Nhan tin hieu dung, dang shutdown toan bo pipeline...")
        capture.stop()
        stop_flag.set()
        for w in window_workers:
            w.stop()
        inference.stop()
        alert_manager.stop()
        time.sleep(1)
        logger.info("Da dung toan bo pipeline. Tam biet!")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ---- Vong lap chinh: in thong ke dinh ky de theo doi suc khoe pipeline ----
    try:
        while True:
            time.sleep(10)
            removed = flow_buffer.cleanup_dead_flows(config.FLOW_TIMEOUT)
            logger.info(
                "STAT | packet=%d | dropped=%d | Q1=%d | Q2=%d | alertQ=%d | "
                "active_flows=%d | flows_cleaned=%d",
                capture.packet_count, capture.dropped_count,
                queue1.qsize(), queue2.qsize(), alert_queue.qsize(),
                flow_buffer.stats()["active_flows"], removed,
            )
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()