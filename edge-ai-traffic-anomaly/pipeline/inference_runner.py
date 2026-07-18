"""
inference_runner.py — Inference engine + incremental learning + alert.

Flow readme §0:
  Feature → Scaler + Encoder → Score vs δ → Alert/Log → Incremental μ_norm → loop
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from configs.paths import load_config
from model.inference import ETSSLInference
from pipeline.alert_manager import AlertManager
from pipeline.feature_extractor import FlowRecord
from pipeline.incremental_learner import IncrementalLearner
from pipeline.shared_state import PipelineStateWriter

logger = logging.getLogger(__name__)


class PipelineInferenceRunner:
    """
    Chạy inference cho mỗi flow, gắn alert + incremental learning.
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
        self.engine = engine
        self.learner = learner
        self.alerts = alert_manager or AlertManager()
        self.state = state_writer or PipelineStateWriter()
        self.kappa = kappa
        self.drift_rollback_threshold = drift_rollback_threshold
        self._flows_processed = 0
        self.flow_history = {}  # flow_id -> flow_info dict
        self.recent_high_scores = []  # list of (flow_id, score, features)

    @classmethod
    def from_config(
        cls,
        model_dir: str | Path,
        backend: str = "onnx",
    ) -> "PipelineInferenceRunner":
        cfg = load_config()
        det = cfg["detection"]
        inc = cfg["incremental"]
        log_cfg = cfg["logging"]
        pipe = cfg["pipeline"]

        engine = ETSSLInference(str(model_dir), backend=backend)
        mu_norm = np.load(Path(model_dir) / "mu_norm.npy")
        learner = IncrementalLearner(
            initial_mu=mu_norm,
            alpha=inc["alpha"],
            update_interval=inc["update_interval"],
            min_batch=inc["min_normal_batch"],
            rollback_window=inc["rollback_window"],
            log_dir=log_cfg["log_dir"],
        )
        alerts = AlertManager(
            log_path=log_cfg["alert_log_path"],
            summary_path=log_cfg.get("alert_summary_path"),
        )
        state = PipelineStateWriter(
            state_path=Path(log_cfg["log_dir"]) / "pipeline_state.json"
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
        """Ngưỡng thực tế = δ × κ (readme: κ là bội số nhạy)."""
        return float(self.engine.delta * self.kappa)

    def process_flow(self, flow: FlowRecord, feature_vector: np.ndarray) -> dict:
        """
        Chạy full pipeline cho 1 flow:
        scale → encode → score → alert/log → incremental update μ_norm.
        """
        import time
        from data.feature_schema import MODEL_FEATURES

        result = self.engine.predict(feature_vector)
        score = result["score"]
        is_anomaly = score > self.effective_delta
        result["is_anomaly"] = is_anomaly
        result["effective_delta"] = self.effective_delta

        flow_id = f"{flow.src_ip}:{flow.src_port}->{flow.dst_ip}:{flow.dst_port}"
        src = f"{flow.src_ip}:{flow.src_port}"
        dst = f"{flow.dst_ip}:{flow.dst_port}"

        # -------------------------------------------------------------
        # DIAGNOSTIC #1: Check for duplicate flow IDs in a short time
        # -------------------------------------------------------------
        if flow_id in self.flow_history:
            prev = self.flow_history[flow_id]
            time_diff = time.time() - prev["timestamp"]
            # Bug thật (FIN/RST split): time_diff < 0.5s (cùng TCP session)
            # Legitimate timeout split: time_diff > 30s (flow_timeout_sec)
            # Threshold 5s đủ để bắt bug nhưng không báo false alarm từ timeout
            if time_diff < 5.0:
                print("\n" + "=" * 90)
                print(f"⚠️  DUPLICATE FLOW DETECTED FOR flow_id: {flow_id}")
                print(f"   Time diff: {time_diff:.2f}s | Prev Packets: {prev['packets']} | Curr Packets: {flow.packet_count}")
                print(f"   Prev Score: {prev['score']:.4f} | Curr Score: {score:.4f}")
                print("=" * 90)
                print(f"{'Index':<5} {'Feature Name':<35} {'Prev Raw':<15} {'Curr Raw':<15} {'Diff':<15}")
                print("-" * 90)
                for idx, fname in enumerate(MODEL_FEATURES):
                    v_prev = prev["features"][idx]
                    v_curr = feature_vector[idx]
                    print(f"{idx:<5} {fname[:35]:<35} {v_prev:<15.4f} {v_curr:<15.4f} {abs(v_prev-v_curr):<15.4f}")
                print("=" * 90 + "\n")

        # -------------------------------------------------------------
        # DIAGNOSTIC #2: Check for scores clustering in high range
        # -------------------------------------------------------------
        if score > 10000:
            self.recent_high_scores.append((flow_id, score, feature_vector))
            if len(self.recent_high_scores) >= 3:
                print("\n" + "=" * 90)
                print(f"⚠️  SCORE CLUSTERING DETECTED (Multiple high-score flows > 10000)")
                print("=" * 90)
                print(f"{'Index':<5} {'Feature Name':<35}", end="")
                for h_id, h_score, _ in self.recent_high_scores[-3:]:
                    truncated_id = h_id.split("->")[-1]
                    print(f" | {truncated_id[:12]:<12} (S={int(h_score)})", end="")
                print("\n" + "-" * 90)
                for idx, fname in enumerate(MODEL_FEATURES):
                    print(f"{idx:<5} {fname[:35]:<35}", end="")
                    for _, _, h_feat in self.recent_high_scores[-3:]:
                        print(f" | {h_feat[idx]:<19.4f}", end="")
                    print()
                print("=" * 90 + "\n")
                # Keep list reasonable
                self.recent_high_scores = self.recent_high_scores[-5:]

        # Update history
        self.flow_history[flow_id] = {
            "features": feature_vector.copy(),
            "packets": flow.packet_count,
            "score": score,
            "timestamp": time.time()
        }

        self.state.record_flow(
            score=score,
            is_anomaly=is_anomaly,
            src=src,
            dst=dst,
            latency_ms=result["latency_ms"],
        )

        if is_anomaly:
            self.alerts.raise_alert(
                score=score,
                delta=self.effective_delta,
                flow_id=flow_id,
                src=src,
                dst=dst,
                extra={"protocol": flow.protocol, "packets": flow.packet_count},
            )
        else:
            update_info = None
            if self.learner:
                update_info = self.learner.observe(
                    result["embedding"], is_anomaly=False, score=score
                )
                if update_info:
                    drift = update_info["drift"]
                    self.engine.update_mu_norm(self.learner.mu_norm.astype(np.float32))
                    self.state.record_mu_update(drift, update_info["update_id"])
                    if drift > self.drift_rollback_threshold:
                        logger.warning(
                            "μ_norm drift %.4f > threshold — rollback", drift
                        )
                        self.learner.rollback(n_steps=1)
                        self.engine.update_mu_norm(
                            self.learner.mu_norm.astype(np.float32)
                        )

        self._flows_processed += 1
        status = "ANOMALY" if is_anomaly else "NORMAL"
        logger.info(
            "[%s] %s | score=%.2f δ=%.2f",
            status,
            flow_id,
            score,
            self.effective_delta,
        )
        return result
