import threading
import queue
import logging
import numpy as np
import joblib

from .models import Prediction

logger = logging.getLogger("nids.inference")


class InferenceEngine(threading.Thread):
    def __init__(self, in_queue: "queue.Queue", alert_queue: "queue.Queue",
                 model_path, scaler_path, label_encoder_path, feature_list_path,
                 benign_label="BENIGN", vote_mode="weighted",
                 window_weights=None, min_confidence=0.5):
        super().__init__(name="Flow3-Inference", daemon=True)
        self.in_queue = in_queue
        self.alert_queue = alert_queue
        self.benign_label = benign_label
        self.vote_mode = vote_mode
        self.window_weights = window_weights or {}
        self.min_confidence = min_confidence
        self._stop_event = threading.Event()

        logger.info("Dang load model/scaler/label_encoder/feature_list tu disk...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.feature_list = joblib.load(feature_list_path)
        logger.info("Da load xong model. So feature=%d | So class=%d | Classes=%s",
                    len(self.feature_list), len(self.label_encoder.classes_),
                    list(self.label_encoder.classes_))

        # flow_key -> {window_size: (label, confidence, ts)} - de vote giua cac window
        self._recent_predictions = {}
        self._lock = threading.Lock()

    # ---------------- xu ly 1 feature vector ----------------
    def _vectorize(self, feature_values: dict):
        """Sap xep dung thu tu feature_list, dien 0 neu thieu, xu ly inf/nan nhu luc train."""
        row = [feature_values.get(f, 0.0) for f in self.feature_list]
        arr = np.array(row, dtype="float64").reshape(1, -1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def _predict_one(self, arr):
        scaled = self.scaler.transform(arr)
        pred_idx = int(self.model.predict(scaled)[0])
        label = self.label_encoder.inverse_transform([pred_idx])[0]

        confidence = 1.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(scaled)[0]
            confidence = float(proba[pred_idx])
        return label, confidence

    # ---------------- ket hop nhieu window ----------------
    def _combine_votes(self, flow_key):
        with self._lock:
            windows = dict(self._recent_predictions.get(flow_key, {}))
        if not windows:
            return None

        if self.vote_mode == "majority":
            labels = [v[0] for v in windows.values()]
            label = max(set(labels), key=labels.count)
            confs = [v[1] for v in windows.values() if v[0] == label]
            return label, float(np.mean(confs)) if confs else 0.0

        # weighted vote (mac dinh): cua so nao co weight cao hon thi anh huong nhieu hon
        score, weight_sum = {}, {}
        for win_size, (label, conf, _ts) in windows.items():
            w = self.window_weights.get(win_size, 1.0)
            score[label] = score.get(label, 0.0) + w * conf
            weight_sum[label] = weight_sum.get(label, 0.0) + w

        best_label = max(score, key=score.get)
        best_conf = score[best_label] / max(weight_sum[best_label], 1e-9)
        return best_label, float(best_conf)

    # ---------------- vong lap chinh ----------------
    def run(self):
        logger.info("Flow3 (Inference Engine) bat dau.")
        while not self._stop_event.is_set():
            try:
                fv = self.in_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                arr = self._vectorize(fv.values)
                label, confidence = self._predict_one(arr)

                with self._lock:
                    self._recent_predictions.setdefault(fv.flow_key, {})
                    self._recent_predictions[fv.flow_key][fv.window_size] = (
                        label, confidence, fv.ts
                    )

                combined = self._combine_votes(fv.flow_key)
                if combined is None:
                    continue
                final_label, final_conf = combined

                is_attack = final_label != self.benign_label
                # Tan cong nhung confidence qua thap -> khong du tin cay, bo qua de tranh spam
                if is_attack and final_conf < self.min_confidence:
                    continue

                pred = Prediction(
                    flow_key=fv.flow_key,
                    window_size=fv.window_size,
                    ts=fv.ts,
                    label=final_label,
                    confidence=final_conf,
                )
                try:
                    self.alert_queue.put_nowait(pred)
                except queue.Full:
                    logger.warning("Alert queue dang day, bo qua 1 prediction.")

            except Exception:
                logger.exception("Loi trong Flow3 khi xu ly 1 feature vector")

        logger.info("Flow3 (Inference Engine) da dung.")

    def stop(self):
        self._stop_event.set()