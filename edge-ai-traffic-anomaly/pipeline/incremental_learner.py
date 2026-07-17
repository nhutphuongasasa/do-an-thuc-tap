"""
incremental_learner.py — Incremental Learning cho ET-SSL.

Cập nhật tâm cụm μ_norm theo công thức trong bài báo:
    μ_norm^(t+1) = α · μ_norm^(t) + (1-α) · (1/|N|) Σ z_i, i ∈ N

Chỉ cập nhật với flow đã được xác nhận là "normal" (score ≤ δ).
Lưu lịch sử μ_norm để có thể rollback khi phát hiện drift.

Tham chiếu: Giai đoạn 6 trong kế hoạch dự án.
"""

import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
import threading

logger = logging.getLogger(__name__)


class IncrementalLearner:
    """
    Cập nhật μ_norm theo EMA (Exponential Moving Average) trên normal flows.

    Attributes:
        mu_norm: current centroid (embed_dim,)
        alpha: decay factor (0 < α < 1) — giá trị cao → cập nhật chậm
        update_interval: số flow normal cần tích lũy trước khi update
    """

    def __init__(
        self,
        initial_mu: np.ndarray,
        alpha: float = 0.99,
        update_interval: int = 100,
        min_batch: int = 10,
        rollback_window: int = 10,
        log_dir: str = "logs",
    ):
        """
        Args:
            initial_mu: vector μ_norm ban đầu (embed_dim,)
            alpha: decay factor α (default 0.99 từ bài báo)
            update_interval: cập nhật sau N normal flow
            min_batch: số flow tối thiểu để trigger update
            rollback_window: số checkpoint lưu để rollback
            log_dir: thư mục lưu log
        """
        self.mu_norm = initial_mu.copy().astype(np.float64)
        self.alpha = alpha
        self.update_interval = update_interval
        self.min_batch = min_batch
        self.rollback_window = rollback_window
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Buffer tích lũy embedding của normal flows
        self._normal_embeddings: list[np.ndarray] = []
        self._total_updates = 0
        self._total_normal_seen = 0
        self._total_anomaly_seen = 0

        # History để rollback
        self._mu_history: list[tuple[datetime, np.ndarray]] = []
        self._mu_history.append((datetime.now(), self.mu_norm.copy()))

        # Thread safety
        self._lock = threading.Lock()

        # Log file
        self._log_path = self.log_dir / "incremental_log.jsonl"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def observe(
        self,
        embedding: np.ndarray,
        is_anomaly: bool,
        score: float,
    ) -> Optional[dict]:
        """
        Quan sát 1 flow embedding và thêm vào buffer nếu là normal.

        Args:
            embedding: (embed_dim,) numpy array từ encoder
            is_anomaly: kết quả phân loại
            score: anomaly score S(x)

        Returns:
            Update info nếu μ_norm được cập nhật, else None
        """
        with self._lock:
            if is_anomaly:
                self._total_anomaly_seen += 1
                return None

            # Flow normal → thêm vào buffer
            self._normal_embeddings.append(embedding.copy())
            self._total_normal_seen += 1

            # Check nếu đủ để update
            if len(self._normal_embeddings) >= self.update_interval:
                return self._perform_update()

        return None

    def observe_batch(
        self,
        embeddings: np.ndarray,
        is_anomaly: np.ndarray,
        scores: np.ndarray,
    ) -> Optional[dict]:
        """
        Quan sát batch flows.

        Args:
            embeddings: (N, embed_dim)
            is_anomaly: (N,) bool array
            scores: (N,) float array
        """
        update_info = None
        for i in range(len(embeddings)):
            info = self.observe(embeddings[i], bool(is_anomaly[i]), float(scores[i]))
            if info is not None:
                update_info = info
        return update_info

    def force_update(self) -> Optional[dict]:
        """Ép cập nhật μ_norm nếu buffer có đủ min_batch sample."""
        with self._lock:
            if len(self._normal_embeddings) >= self.min_batch:
                return self._perform_update()
        return None

    def rollback(self, n_steps: int = 1) -> bool:
        """
        Rollback μ_norm về checkpoint trước đó.

        Args:
            n_steps: số bước rollback (default 1 = bước gần nhất)

        Returns:
            True nếu rollback thành công
        """
        with self._lock:
            if len(self._mu_history) <= n_steps:
                logger.warning(f"Cannot rollback {n_steps} steps — history too short")
                return False

            target_idx = len(self._mu_history) - 1 - n_steps
            ts, old_mu = self._mu_history[target_idx]
            logger.info(
                f"Rollback to checkpoint at {ts.isoformat()} "
                f"(|Δμ| = {np.linalg.norm(self.mu_norm - old_mu):.4f})"
            )
            self.mu_norm = old_mu.copy()
            # Xóa các checkpoint sau rollback
            self._mu_history = self._mu_history[:target_idx + 1]
            return True

    @property
    def stats(self) -> dict:
        """Thống kê hiện tại."""
        with self._lock:
            return {
                "total_updates": self._total_updates,
                "total_normal_seen": self._total_normal_seen,
                "total_anomaly_seen": self._total_anomaly_seen,
                "buffer_size": len(self._normal_embeddings),
                "history_size": len(self._mu_history),
                "mu_norm_mean": float(np.mean(self.mu_norm)),
                "mu_norm_std": float(np.std(self.mu_norm)),
            }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------
    def _perform_update(self) -> dict:
        """
        Thực hiện cập nhật EMA:
        μ^(t+1) = α·μ^(t) + (1-α)·mean(z_i)
        """
        batch = np.stack(self._normal_embeddings, axis=0)  # (N, D)
        batch_mean = batch.mean(axis=0)                    # (D,)

        old_mu = self.mu_norm.copy()
        self.mu_norm = self.alpha * self.mu_norm + (1 - self.alpha) * batch_mean
        self._total_updates += 1

        drift = float(np.linalg.norm(self.mu_norm - old_mu))

        # Lưu checkpoint
        ts = datetime.now()
        self._mu_history.append((ts, self.mu_norm.copy()))
        # Giới hạn lịch sử
        if len(self._mu_history) > self.rollback_window + 1:
            self._mu_history = self._mu_history[-(self.rollback_window + 1):]

        # Xóa buffer
        n_used = len(self._normal_embeddings)
        self._normal_embeddings = []

        update_info = {
            "update_id": self._total_updates,
            "timestamp": ts.isoformat(),
            "n_flows_used": n_used,
            "drift": round(drift, 6),
            "alpha": self.alpha,
        }

        # Log
        self._log_update(update_info)

        logger.info(
            f"μ_norm update #{self._total_updates}: "
            f"n={n_used}, drift={drift:.4f}"
        )
        return update_info

    def _log_update(self, info: dict):
        """Ghi log update vào JSONL file."""
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(info) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write log: {e}")

    def save_mu_norm(self, path: str | Path):
        """Lưu μ_norm hiện tại ra file."""
        np.save(path, self.mu_norm.astype(np.float32))

    def get_mu_norm_history(self) -> list[dict]:
        """Trả về lịch sử μ_norm theo thời gian."""
        with self._lock:
            return [
                {
                    "timestamp": ts.isoformat(),
                    "mu_mean": float(np.mean(mu)),
                    "mu_norm": float(np.linalg.norm(mu)),
                }
                for ts, mu in self._mu_history
            ]


# =====================================================================
# Drift Simulation (Giai đoạn 6 — test incremental learning)
# =====================================================================
def simulate_drift(
    engine,
    initial_mu: np.ndarray,
    n_epochs: int = 5,
    n_flows_per_epoch: int = 1000,
    alpha: float = 0.99,
    anomaly_ratio: float = 0.15,
    drift_magnitude: float = 1.0,
) -> dict:
    """
    Giả lập traffic drift theo thời gian và test incremental learning.

    Mỗi "epoch" = 1 đợt traffic với phân phối dịch nhẹ (drift).
    → Đo model có thích nghi μ_norm hay không.

    Returns:
        dict với drift history và detection metrics theo epoch
    """
    from sklearn.preprocessing import StandardScaler

    learner = IncrementalLearner(initial_mu=initial_mu, alpha=alpha)
    rng = np.random.default_rng(42)

    history = []
    embed_dim = len(initial_mu)
    input_dim = 20

    print("\n🌊 Drift Simulation:")
    print(f"  {n_epochs} epochs × {n_flows_per_epoch} flows/epoch")
    print(f"  Drift magnitude: {drift_magnitude} | α={alpha}")
    print(f"  {'Epoch':<8} {'Drift':>8} {'Normal%':>9} {'Anomaly%':>10} {'μ drift':>9}")
    print(f"  {'─'*50}")

    # Cần scaler — nếu engine không có, dùng identity
    scaler = engine.scaler

    for epoch in range(n_epochs):
        # Tăng dần drift theo epoch
        current_drift = drift_magnitude * (epoch / max(n_epochs - 1, 1))

        n_attack = int(n_flows_per_epoch * anomaly_ratio)
        n_normal = n_flows_per_epoch - n_attack

        # Traffic normal — dịch chuyển dần (drift)
        X_normal = rng.normal(current_drift, 1.0, (n_normal, input_dim)).astype(np.float32)
        # Traffic attack — luôn lệch xa hơn
        X_attack = rng.normal(current_drift + 3.0, 1.5, (n_attack, input_dim)).astype(np.float32)

        X_epoch = np.vstack([X_normal, X_attack])
        y_epoch = np.array([0] * n_normal + [1] * n_attack)

        # Shuffle
        idx = rng.permutation(len(X_epoch))
        X_epoch, y_epoch = X_epoch[idx], y_epoch[idx]

        # Scale nếu có
        if scaler:
            X_epoch_s = scaler.transform(X_epoch).astype(np.float32)
        else:
            X_epoch_s = X_epoch

        # Inference + observe
        old_mu = learner.mu_norm.copy()
        preds_raw = engine.predict_batch(X_epoch_s)

        embeddings = np.array([r["embedding"] for r in preds_raw])
        scores     = np.array([r["score"] for r in preds_raw])
        is_anomaly = np.array([r["is_anomaly"] for r in preds_raw])

        # Update engine's mu_norm trong learner
        for i in range(len(embeddings)):
            learner.observe(embeddings[i], bool(is_anomaly[i]), float(scores[i]))

        # Sync μ_norm về engine
        engine.update_mu_norm(learner.mu_norm.astype(np.float32))

        # Tính metrics
        from evaluation.metrics import compute_binary_metrics
        preds_bin = is_anomaly.astype(int)
        m = compute_binary_metrics(y_epoch, preds_bin)
        mu_drift = float(np.linalg.norm(learner.mu_norm - old_mu))

        epoch_info = {
            "epoch": epoch + 1,
            "traffic_drift": round(current_drift, 2),
            "mu_drift": round(mu_drift, 6),
            "learner_stats": learner.stats,
            **{f"det_{k}": v for k, v in m.items() if not isinstance(v, int)},
        }
        history.append(epoch_info)

        pct_detected = m["recall"] * 100
        print(f"  Epoch {epoch+1:<4} {current_drift:>8.2f} {m['accuracy']*100:>9.2f}% "
              f"{pct_detected:>10.2f}% {mu_drift:>9.6f}")

    print(f"\n  ✅ Drift simulation complete!")
    print(f"     μ_norm total drift: "
          f"{np.linalg.norm(learner.mu_norm - initial_mu):.4f}")

    return {"history": history, "final_stats": learner.stats}


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    MODEL_DIR = str(
        Path(__file__).parent.parent.parent / "TrafficGuard/models/edge_ai-20260716T101644Z-1-001/edge_ai"
    )

    # Load initial μ_norm
    mu_path = Path(MODEL_DIR) / "mu_norm.npy"
    initial_mu = np.load(mu_path)
    print(f"📊 Initial μ_norm: shape={initial_mu.shape}, mean={initial_mu.mean():.4f}")

    learner = IncrementalLearner(
        initial_mu=initial_mu,
        alpha=0.99,
        update_interval=50,
    )

    # Giả lập 200 normal flows
    print("\n🔄 Simulating 200 normal flows + 50 anomalies...")
    rng = np.random.default_rng(42)

    for i in range(200):
        z_normal = rng.normal(0, 0.5, initial_mu.shape).astype(np.float32)
        learner.observe(z_normal, is_anomaly=False, score=10.0)

    for i in range(50):
        z_anom = rng.normal(5, 1.0, initial_mu.shape).astype(np.float32)
        learner.observe(z_anom, is_anomaly=True, score=200.0)

    # Force final update
    learner.force_update()

    stats = learner.stats
    print(f"\n📊 Learner stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    drift = np.linalg.norm(learner.mu_norm - initial_mu)
    print(f"\n  μ_norm total drift: {drift:.6f}")
    print("✅ IncrementalLearner test complete")
