import threading
import queue
import logging
import numpy as np
import pandas as pd
import joblib

from dto import Prediction

logger = logging.getLogger("nids.inference")


class InferenceEngine(threading.Thread):
    def __init__(self, in_queue: "queue.Queue", alert_queue: "queue.Queue",
                 model_path, scaler_path, label_encoder_path, feature_list_path,
                 benign_label="BENIGN", vote_mode="weighted",
                 window_weights=None, min_confidence=0.6,
                 required_window_sizes=None,   # vd {1, 3, 5} -> phai co DU tat ca window nay moi vote
                 min_windows=2,                # hoac: can it nhat bao nhieu window (dung khi required_window_sizes=None)
                 confirm_streak=2,             # phai lien tiep bi danh gia la attack bao nhieu lan moi bao dong that
                 stale_window_seconds=15.0):   # window nao cu qua thi loai khoi vote (tranh dung du lieu "chet")
        super().__init__(name="Flow3-Inference", daemon=True)
        self.in_queue = in_queue
        self.alert_queue = alert_queue
        self.benign_label = benign_label
        self.vote_mode = vote_mode

        self.window_weights = window_weights or {}
        self.min_confidence = min_confidence

        self.required_window_sizes = set(required_window_sizes) if required_window_sizes else None
        self.min_windows = min_windows
        self.confirm_streak = confirm_streak
        self.stale_window_seconds = stale_window_seconds

        self._stop_event = threading.Event()

        logger.info("Dang load model/scaler/label_encoder/feature_list tu disk...")
        self.model = joblib.load(model_path)

        # Toi uu: RandomForest mac dinh chay song song n_jobs=16 (tuy config luc train).
        # Voi inference realtime, moi lan chi predict 1 dong du lieu nen chay song song
        # gay overhead tao thread pool moi lan goi predict/predict_proba, lam cham va
        # spam log "[Parallel(n_jobs=...)]". Ep ve n_jobs=1 de predict nhanh va gon hon.
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1
        if hasattr(self.model, "verbose"):
            self.model.verbose = 0

        self.scaler = joblib.load(scaler_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.feature_list = joblib.load(feature_list_path)
        logger.info("Da load xong model. So feature=%d | So class=%d | Classes=%s",
                    len(self.feature_list), len(self.label_encoder.classes_),
                    list(self.label_encoder.classes_))

        self._recent_predictions = {}
        self._attack_streak = {}
        self._lock = threading.Lock()

    #nhan feature tu flow 2 chuan hoa no thanh array so de model du doan 
    def _vectorize(self, feature_values: dict):
        row = [feature_values.get(f, 0.0) for f in self.feature_list]
        arr = np.array(row, dtype="float64").reshape(1, -1)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    #dua vao model va nhan ket qua loai tan cong va do tu tin cay cua model
    def _predict_one(self, arr):
        # Toi uu: chuyen array thanh DataFrame co ten cot dung nhu luc train,
        # tranh warning "X does not have valid feature names" va giam rui ro
        # sai lech neu thu tu cot bi dao lon o dau do trong code.
        arr_df = pd.DataFrame(arr, columns=self.feature_list)
        scaled = self.scaler.transform(arr_df)

        pred_idx = int(self.model.predict(scaled)[0])
        label = self.label_encoder.inverse_transform([pred_idx])[0]

        confidence = 1.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(scaled)[0]
            confidence = float(proba[pred_idx])
        return label, confidence

    def _combine_votes(self, flow_key, now_ts):
        with self._lock:
            windows = dict(self._recent_predictions.get(flow_key, {}))

        if not windows:
            return None

        windows = {
            w: v for w, v in windows.items()
            if now_ts - v[2] <= self.stale_window_seconds
        }
        if not windows:
            return None

        if self.required_window_sizes is not None:
            if not self.required_window_sizes.issubset(windows.keys()):
                return None  # chua du cac window can thiet -> chua ket luan, cho them
        else:
            if len(windows) < self.min_windows:
                return None  # chua du so luong window toi thieu -> chua ket luan

        if self.vote_mode == "majority":
            labels = [v[0] for v in windows.values()]
            label = max(set(labels), key=labels.count)
            confs = [v[1] for v in windows.values() if v[0] == label]
            return label, float(np.mean(confs)) if confs else 0.0

        score, weight_sum = {}, {}
        for win_size, (label, conf, _ts) in windows.items():
            w = self.window_weights.get(win_size, float(win_size))
            score[label] = score.get(label, 0.0) + w * conf
            weight_sum[label] = weight_sum.get(label, 0.0) + w

        best_label = max(score, key=score.get)
        best_conf = score[best_label] / max(weight_sum[best_label], 1e-9)
        return best_label, float(best_conf)

    def run(self):
        logger.info("Flow3 (Inference Engine) bat dau. "
                     "required_windows=%s | min_windows=%s | confirm_streak=%s",
                     self.required_window_sizes, self.min_windows, self.confirm_streak)
        while not self._stop_event.is_set():
            try:
                fv = self.in_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                arr = self._vectorize(fv.values)
                label, confidence = self._predict_one(arr)
                logger.debug(
                    "Flow %s | window=%ds | predict=%s (conf=%.2f)",
                    fv.flow_key, fv.window_size, label, confidence,
                )

                #luu ket qua du doan vao _recent_predictions de dung cho viec vote
                with self._lock:
                    self._recent_predictions.setdefault(fv.flow_key, {})
                    self._recent_predictions[fv.flow_key][fv.window_size] = (
                        label, confidence, fv.ts
                    )

                combined = self._combine_votes(fv.flow_key, now_ts=fv.ts)
                if combined is None:
                    continue
                final_label, final_conf = combined
                logger.debug(
                    "Flow %s | VOTE ket qua: %s (conf=%.2f)",
                    fv.flow_key, final_label, final_conf,
                )

                is_attack = final_label != self.benign_label

                if is_attack and final_conf < self.min_confidence:
                    with self._lock:
                        self._attack_streak[fv.flow_key] = 0
                    continue

                with self._lock:
                    if is_attack:
                        self._attack_streak[fv.flow_key] = self._attack_streak.get(fv.flow_key, 0) + 1
                    else:
                        self._attack_streak[fv.flow_key] = 0
                    streak = self._attack_streak[fv.flow_key]

                if is_attack and streak < self.confirm_streak:
                    logger.debug(
                        "Flow %s: nghi ngo '%s' (streak=%d/%d, conf=%.2f) - cho xac nhan them",
                        fv.flow_key, final_label, streak, self.confirm_streak, final_conf,
                    )
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