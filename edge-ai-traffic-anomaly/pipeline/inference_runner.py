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

    def process_flow(self, flow: FlowRecord, feature_vector: np.ndarray) -> Dict[str, Any]:
        """
        Xử lý quy trình toàn diện cho một luồng đơn lẻ.

        Args:
            flow: Bản ghi đối tượng của luồng.
            feature_vector: Vector trích xuất đặc trưng gồm 20 chiều.

        Returns:
            Từ điển mô tả kết quả chuẩn đoán.
        """
        result = self._run_inference(feature_vector)
        score = result["score"]
        is_anomaly = result["is_anomaly"]
        
        flow_id = f"{flow.src_ip}:{flow.src_port}->{flow.dst_ip}:{flow.dst_port}"
        src = f"{flow.src_ip}:{flow.src_port}"
        dst = f"{flow.dst_ip}:{flow.dst_port}"

        if logger.isEnabledFor(logging.DEBUG):
            self._run_diagnostics(flow, feature_vector, flow_id, score)

        # Cập nhật tiến độ trực quan
        self.state.record_flow(
            score=score,
            is_anomaly=is_anomaly,
            src=src,
            dst=dst,
            latency_ms=result["latency_ms"],
        )

        if is_anomaly:
            self._handle_anomaly(flow, flow_id, src, dst, score)
        else:
            self._handle_normal(result["embedding"], score)

        self._flows_processed += 1
        status = "ANOMALY" if is_anomaly else "NORMAL"
        logger.info("[%s] %s | score=%.2f δ=%.2f", status, flow_id, score, self.effective_delta)
        
        return result

    def _run_inference(self, feature_vector: np.ndarray) -> Dict[str, Any]:
        """Tính toán vector nhúng (embedding) và mức độ bất thường."""
        result = self.engine.predict(feature_vector)
        score = float(result["score"])
        result["is_anomaly"] = score > self.effective_delta
        result["effective_delta"] = self.effective_delta
        return result

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
