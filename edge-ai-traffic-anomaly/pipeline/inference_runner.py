"""
inference_runner.py — Trình điều phối luồng suy luận (Inference Engine Orchestrator).

Chịu trách nhiệm kết nối các khâu trong quy trình chuẩn đoán cho mỗi luồng mạng:
1. Chuẩn hóa & Sinh nhúng đặc trưng
2. So sánh điểm số S(x) với ngưỡng Δ để phát hiện bất thường
3. Phân loại và Cảnh báo
4. Cập nhật học gia tăng (Incremental Learning) cho tâm cụm μ_norm
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import queue
import threading
import numpy as np

from configs.paths import load_config
from data.feature_schema import MODEL_FEATURES
from model.inference import ETSSLInference
from pipeline.alert_manager import AlertManager
from pipeline.feature_extractor import FlowRecord
from pipeline.incremental_learner import IncrementalLearner
from pipeline.shared_state import PipelineStateWriter

logger = logging.getLogger(__name__)


class PipelineInferenceRunner:
    """
    Trình điều phối (Orchestrator) thực thi toàn bộ quy trình phát hiện bất thường
    trên từng luồng mạng, kết nối Engine, Learner và AlertManager.
    """

    def __init__(
        self,
        engine: ETSSLInference,
        *,
        learner: IncrementalLearner | None = None,
        alert_manager: AlertManager | None = None,
        state_writer: PipelineStateWriter | None = None,
        kappa: float = 1.0,
        drift_rollback_threshold: float = 5.0,
    ):
        """
        Khởi tạo Trình điều phối.

        Args:
            engine: Khối máy học suy luận.
            learner: Khối học gia tăng (Incremental Learner).
            alert_manager: Khối quản lý cảnh báo.
            state_writer: Khối đồng bộ trạng thái luồng dữ liệu.
            kappa: Bội số cấu hình độ nhạy của ngưỡng (Δ = δ * κ).
            drift_rollback_threshold: Mức độ trôi dạt (drift) tối đa trước khi thu hồi mô hình.
        """
        self.engine = engine
        self.learner = learner
        self.alerts = alert_manager or AlertManager()
        self.state = state_writer or PipelineStateWriter()
        self.kappa = kappa
        self.drift_rollback_threshold = drift_rollback_threshold
        self._flows_processed = 0
        
        # Biến trạng thái chẩn đoán (Diagnostic states)
        self.flow_history: Dict[str, Dict[str, Any]] = {}
        self.recent_high_scores: List[Tuple[str, float, np.ndarray]] = []

        # Hàng đợi (Queue) cho Inference bất đồng bộ
        self.inference_queue = queue.Queue(maxsize=10000)
        self.batch_size = 32
        self.batch_timeout = 0.05  # 50ms
        self.stop_event = threading.Event()
        self.inference_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.monitor_thread = threading.Thread(target=self._throughput_monitor, daemon=True)
        self.inference_thread.start()
        self.monitor_thread.start()

    def stop(self):
        """Dừng các luồng nền."""
        self.stop_event.set()
        self.inference_thread.join(timeout=2.0)
        self.monitor_thread.join(timeout=2.0)

    @classmethod
    def from_config(
        cls,
        model_dir: str | Path,
        backend: str = "onnx",
    ) -> PipelineInferenceRunner:
        """
        Factory Method: Khởi tạo Trình điều phối dựa trên các tệp cấu hình hệ thống.
        """
        cfg = load_config()
        det = cfg.get("detection", {})
        inc = cfg.get("incremental", {})
        log_cfg = cfg.get("logging", {})

        engine = ETSSLInference(str(model_dir), backend=backend)
        mu_norm_path = Path(model_dir) / "mu_norm.npy"
        
        if not mu_norm_path.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp tâm cụm: {mu_norm_path}")
            
        mu_norm = np.load(mu_norm_path)
        
        learner = IncrementalLearner(
            initial_mu=mu_norm,
            alpha=inc.get("alpha", 0.05),
            update_interval=inc.get("update_interval", 100),
            min_batch=inc.get("min_normal_batch", 20),
            rollback_window=inc.get("rollback_window", 5),
            log_dir=log_cfg.get("log_dir", "logs"),
        )
        
        alerts = AlertManager(
            log_path=log_cfg.get("alert_log_path", "logs/alerts.jsonl"),
            summary_path=log_cfg.get("alert_summary_path"),
        )
        
        state = PipelineStateWriter(
            state_path=Path(log_cfg.get("log_dir", "logs")) / "pipeline_state.json"
        )
        
        return cls(
            engine=engine,
            learner=learner,
            alert_manager=alerts,
            state_writer=state,
            kappa=det.get("kappa", 1.0),
            drift_rollback_threshold=inc.get("drift_rollback_threshold", 5.0),
        )

    @property
    def effective_delta(self) -> float:
        """Tính toán ngưỡng phát hiện thực tế: δ × κ."""
        return float(self.engine.delta * self.kappa)

    def process_flow(self, flow: FlowRecord, feature_vector: np.ndarray):
        """Được gọi bởi Capture layer (hoặc Preprocess worker) để đưa dữ liệu vào hàng đợi (decoupling)."""
        try:
            self.inference_queue.put_nowait((flow, feature_vector))
        except queue.Full:
            logger.warning("WARNING: Hàng đợi inference_queue đã đầy (maxsize=%d)! Đang rớt gói.", self.inference_queue.maxsize)

    def _batch_worker(self):
        """Worker thread: gom các luồng thành 1 batch (theo size hoặc timeout) rồi gọi predict_batch."""
        batch_flows = []
        batch_features = []
        last_process_time = time.time()

        while not self.stop_event.is_set() or not self.inference_queue.empty():
            try:
                # Đợi dữ liệu với timeout nhỏ để kiểm tra xem có cần xử lý batch hiện tại không
                flow, feat = self.inference_queue.get(timeout=0.01)
                batch_flows.append(flow)
                batch_features.append(feat)
                self.inference_queue.task_done()
            except queue.Empty:
                pass

            now = time.time()
            if len(batch_flows) >= self.batch_size or (len(batch_flows) > 0 and (now - last_process_time) >= self.batch_timeout) or self.stop_event.is_set():
                if len(batch_flows) > 0:
                    self._process_batch(batch_flows, batch_features)
                    batch_flows.clear()
                    batch_features.clear()
                    last_process_time = time.time()

    def _process_batch(self, flows: List[FlowRecord], features: List[np.ndarray]):
        """Xử lý thực sự cho một lô dữ liệu."""
        X = np.vstack(features)
        
        # Gọi engine suy luận lô
        results = self.engine.predict_batch(X)
        
        for i, res in enumerate(results):
            flow = flows[i]
            feat = features[i]
            score = res["score"]
            is_anomaly = res["is_anomaly"]
            
            flow_id = f"{flow.src_ip}:{flow.src_port}->{flow.dst_ip}:{flow.dst_port}"
            src = f"{flow.src_ip}:{flow.src_port}"
            dst = f"{flow.dst_ip}:{flow.dst_port}"

            if logger.isEnabledFor(logging.DEBUG):
                self._run_diagnostics(flow, feat, flow_id, score)

            # Cập nhật trạng thái
            self.state.record_flow(
                score=score,
                is_anomaly=is_anomaly,
                src=src,
                dst=dst,
                latency_ms=res["latency_ms"],
            )

            if is_anomaly:
                self._handle_anomaly(flow, flow_id, src, dst, score)
            else:
                self._handle_normal(res["embedding"], score)

            self._flows_processed += 1
            if is_anomaly:
                logger.info("[ANOMALY] %s | score=%.2f δ=%.2f", flow_id, score, self.effective_delta)
            else:
                logger.debug("[NORMAL] %s | score=%.2f δ=%.2f", flow_id, score, self.effective_delta)

    def _throughput_monitor(self):
        """Worker thread: Theo dõi và in ra throughput mỗi 5 giây."""
        last_time = time.time()
        last_count = 0
        while not self.stop_event.is_set():
            time.sleep(5.0)
            now = time.time()
            current_count = self._flows_processed
            delta_count = current_count - last_count
            delta_time = now - last_time
            if delta_count > 0:
                tps = delta_count / delta_time
                logger.info("[MONITOR] Real-time Throughput: %.2f flows/sec (Total: %d)", tps, current_count)
            last_time = now
            last_count = current_count



    def _handle_anomaly(
        self, flow: FlowRecord, flow_id: str, src: str, dst: str, score: float
    ) -> None:
        """Ghi nhận cảnh báo luồng mạng dị thường."""
        self.alerts.raise_alert(
            score=score,
            delta=self.effective_delta,
            flow_id=flow_id,
            src=src,
            dst=dst,
            extra={"protocol": flow.protocol, "packets": flow.packet_count},
        )

    def _handle_normal(self, embedding: np.ndarray, score: float) -> None:
        """Xử lý quy trình học gia tăng cho luồng bình thường, điều chỉnh tâm cụm."""
        if not self.learner:
            return
            
        update_info = self.learner.observe(embedding, is_anomaly=False, score=score)
        
        if update_info:
            drift = update_info["drift"]
            self.engine.update_mu_norm(self.learner.mu_norm.astype(np.float32))
            self.state.record_mu_update(drift, update_info["update_id"])
            
            if drift > self.drift_rollback_threshold:
                logger.warning("μ_norm drift %.4f vượt ngưỡng %s — kích hoạt rollback", drift, self.drift_rollback_threshold)
                self.learner.rollback(n_steps=1)
                self.engine.update_mu_norm(self.learner.mu_norm.astype(np.float32))

    def _run_diagnostics(
        self, flow: FlowRecord, feature_vector: np.ndarray, flow_id: str, score: float
    ) -> None:
        """
        Khối mã phân tích gỡ lỗi (Diagnostics) không hoạt động ở môi trường thực tế mặc định.
        Được dùng để phát hiện sự phân tách luồng không chính xác hoặc dồn cục điểm số lỗi.
        """
        current_time = time.time()
        
        # -------------------------------------------------------------
        # DIAGNOSTIC #1: Phát hiện trùng lặp luồng trong khung thời gian ngắn
        # -------------------------------------------------------------
        if flow_id in self.flow_history:
            prev = self.flow_history[flow_id]
            time_diff = current_time - prev["timestamp"]
            if time_diff < 5.0:
                msg = [
                    "=" * 90,
                    f"⚠️  PHÁT HIỆN TRÙNG LẶP CHO flow_id: {flow_id}",
                    f"   Cách biệt: {time_diff:.2f}s | Gói cũ: {prev['packets']} | Gói mới: {flow.packet_count}",
                    f"   Điểm cũ: {prev['score']:.4f} | Điểm mới: {score:.4f}",
                    "=" * 90,
                    f"{'Index':<5} {'Đặc trưng':<35} {'Cũ':<15} {'Mới':<15} {'Lệch':<15}",
                    "-" * 90
                ]
                for idx, fname in enumerate(MODEL_FEATURES):
                    v_prev = prev["features"][idx]
                    v_curr = feature_vector[idx]
                    msg.append(f"{idx:<5} {fname[:35]:<35} {v_prev:<15.4f} {v_curr:<15.4f} {abs(v_prev-v_curr):<15.4f}")
                msg.append("=" * 90)
                logger.debug("\n".join(msg))

        # Cập nhật lịch sử cho diagnostic #1
        self.flow_history[flow_id] = {
            "features": feature_vector.copy(),
            "packets": flow.packet_count,
            "score": score,
            "timestamp": current_time
        }

        # -------------------------------------------------------------
        # DIAGNOSTIC #2: Phân tích dồn cục điểm số bất thường mức cao
        # -------------------------------------------------------------
        if score > 10000:
            self.recent_high_scores.append((flow_id, score, feature_vector.copy()))
            if len(self.recent_high_scores) >= 3:
                msg = [
                    "=" * 90,
                    "⚠️  PHÁT HIỆN DỒN CỤC ĐIỂM SỐ (Nhiều luồng > 10000)",
                    "=" * 90,
                ]
                
                header = f"{'Index':<5} {'Đặc trưng':<35}"
                for h_id, h_score, _ in self.recent_high_scores[-3:]:
                    truncated_id = h_id.split("->")[-1]
                    header += f" | {truncated_id[:12]:<12} (S={int(h_score)})"
                msg.append(header)
                msg.append("-" * 90)
                
                for idx, fname in enumerate(MODEL_FEATURES):
                    row = f"{idx:<5} {fname[:35]:<35}"
                    for _, _, h_feat in self.recent_high_scores[-3:]:
                        row += f" | {h_feat[idx]:<19.4f}"
                    msg.append(row)
                    
                msg.append("=" * 90)
                logger.debug("\n".join(msg))
                
                # Cắt gọn bộ nhớ
                self.recent_high_scores = self.recent_high_scores[-5:]
