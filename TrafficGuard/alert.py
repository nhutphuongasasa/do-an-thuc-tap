import threading
import queue
import time
import logging
import os
import csv

logger = logging.getLogger("nids.alert")

try:
    import requests
except ImportError:
    requests = None


class AlertManager(threading.Thread):
    def __init__(self, in_queue: "queue.Queue", log_dir: str, alert_csv_path: str,
                 benign_label="BENIGN", cooldown_seconds: float = 5.0,
                 telegram_token: str = "", telegram_chat_id: str = ""):
        super().__init__(name="Flow4-Alert", daemon=True)
        self.in_queue = in_queue
        self.benign_label = benign_label
        self.alert_csv_path = alert_csv_path
        self.cooldown_seconds = cooldown_seconds
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self._stop_event = threading.Event()

        os.makedirs(log_dir, exist_ok=True)
        self._init_csv()

        self._last_alert_ts = {}  # flow_key -> ts lan gui alert gan nhat (chong spam)

        if telegram_token and telegram_chat_id and requests is None:
            logger.warning("Da cau hinh Telegram nhung chua cai thu vien 'requests'. "
                            "Chay: pip install requests")

    def _init_csv(self):
        is_new = not os.path.exists(self.alert_csv_path)
        self._csv_file = open(self.alert_csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        if is_new:
            self._csv_writer.writerow([
                "timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "protocol",
                "window_size", "label", "confidence",
            ])
            self._csv_file.flush()

    def run(self):
        logger.info("Flow4 (Alert Manager) bat dau.")
        while not self._stop_event.is_set():
            try:
                pred = self.in_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            src_ip, src_port, dst_ip, dst_port, proto = pred.flow_key
            is_attack = pred.label != self.benign_label

            self._csv_writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(pred.ts)),
                src_ip, src_port, dst_ip, dst_port, proto,
                pred.window_size, pred.label, f"{pred.confidence:.4f}",
            ])
            self._csv_file.flush()

            if is_attack:
                msg = (f"🚨 [{pred.label}] {src_ip}:{src_port} -> {dst_ip}:{dst_port} "
                       f"({proto}) | window={pred.window_size}s | conf={pred.confidence:.2f}")
                logger.warning(msg)
                if self._should_send_telegram(pred.flow_key):
                    self._send_telegram(msg)
            else:
                logger.debug("Benign: %s:%s -> %s:%s", src_ip, src_port, dst_ip, dst_port)

        self._csv_file.close()
        logger.info("Flow4 (Alert Manager) da dung.")

    def stop(self):
        self._stop_event.set()