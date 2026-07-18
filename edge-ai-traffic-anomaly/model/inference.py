"""
inference.py — ET-SSL Inference Engine

Hàm suy luận thuần: score = ||f_θ(x) - μ_norm||²
Hỗ trợ 3 backend: PyTorch FP32, PyTorch INT8, ONNX Runtime (CPU).
Không phụ thuộc vào training code.

Tham chiếu: Sattar et al. 2025, mục "Anomaly Detection"
"""

import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Optional, Literal, Union
import joblib

logger = logging.getLogger(__name__)


# =====================================================================
# Backend: PyTorch
# =====================================================================
class _TorchBackend:
    """PyTorch inference backend (FP32 hoặc INT8)."""

    def __init__(self, model_dir: Path, use_int8: bool = False):
        import torch
        from model.encoder import load_encoder

        fname = "encoder_int8.pt" if use_int8 else "encoder_fp32.pt"
        self.model = load_encoder(
            weights_path=model_dir / fname,
            config_path=model_dir / "config.json",
            device="cpu",
        )
        self._torch = torch
        self.backend_name = "PyTorch-INT8" if use_int8 else "PyTorch-FP32"

    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (N, input_dim) numpy array đã scale
        Returns:
            z: (N, embed_dim) numpy embeddings
        """
        tensor = self._torch.from_numpy(x.astype(np.float32))
        with self._torch.no_grad():
            z = self.model(tensor).numpy()
        return z


# =====================================================================
# Backend: ONNX Runtime
# =====================================================================
class _OnnxBackend:
    """ONNX Runtime inference backend — CPU only, phù hợp edge."""

    def __init__(self, model_dir: Path):
        import onnxruntime as ort

        onnx_path = model_dir / "encoder_v5.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1  # giả lập edge CPU

        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.backend_name = "ONNX-CPU"

    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x.astype(np.float32)})[0]


# =====================================================================
# ET-SSL Inference Engine (public interface)
# =====================================================================
class ETSSLInference:
    """
    Inference engine cho ET-SSL anomaly detection.

    Anomaly score: S(x) = ||f_θ(x_scaled) - μ_norm||²
    Detection: anomaly nếu S(x) > δ

    Usage:
        engine = ETSSLInference(model_dir="path/to/model", backend="onnx")
        result = engine.predict(feature_vector)
        print(result["is_anomaly"], result["score"])
    """

    BACKENDS = ("fp32", "int8", "onnx")

    def __init__(
        self,
        model_dir: str,
        backend: Literal["fp32", "int8", "onnx"] = "onnx",
        delta_override: Optional[float] = None,
    ):
        """
        Args:
            model_dir: Thư mục chứa model artifacts (encoder*.pt, mu_norm.npy, ...)
            backend: "fp32" | "int8" | "onnx" — backend inference
            delta_override: Ghi đè ngưỡng δ (dùng khi tinh chỉnh trên val set)
        """
        self.model_dir = Path(model_dir)
        self._validate_dir()

        # Load config
        with open(self.model_dir / "config.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.input_dim = self.config["input_dim"]
        self.embed_dim = self.config["embed_dim"]

        # Load μ_norm (tâm cụm bình thường)
        self.mu_norm = np.load(self.model_dir / "mu_norm.npy")  # (embed_dim,)

        # Load δ (ngưỡng anomaly)
        delta_raw = np.load(self.model_dir / "delta.npy")
        self.delta = float(delta_override or delta_raw.flat[0])

        # Load scaler (nếu có)
        scaler_path = self.model_dir / "scaler.pkl"
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        if self.scaler is None:
            logger.warning(
                "scaler.pkl not found in model_dir. "
                "Feature input sẽ được dùng trực tiếp (không scale). "
                "Hãy chắc chắn input đã được chuẩn hóa bên ngoài."
            )

        # Khởi tạo backend
        self.backend = self._init_backend(backend)
        logger.info(
            f"ETSSLInference ready | backend={self.backend.backend_name} "
            f"| δ={self.delta:.4f} | input_dim={self.input_dim}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> dict:
        """
        Predict anomaly score cho 1 sample.

        Args:
            x: (input_dim,) raw feature vector (sẽ được scale tự động nếu có scaler)

        Returns:
            dict với keys: score, is_anomaly, embedding, latency_ms
        """
        t0 = time.perf_counter()

        x_scaled = self._scale(x.reshape(1, -1))                  # (1, input_dim)
        z = self.backend.predict_batch(x_scaled)                    # (1, embed_dim)
        score = float(np.sum((z[0] - self.mu_norm) ** 2))

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "score": score,
            "is_anomaly": score > self.delta,
            "embedding": z[0],
            "latency_ms": latency_ms,
        }

    def predict_batch(self, X: np.ndarray) -> list[dict]:
        """
        Predict cho nhiều sample.

        Args:
            X: (N, input_dim) numpy array

        Returns:
            List[dict] — mỗi phần tử giống output của predict()
        """
        t0 = time.perf_counter()

        X_scaled = self._scale(X)
        Z = self.backend.predict_batch(X_scaled)                   # (N, embed_dim)
        diff = Z - self.mu_norm[np.newaxis, :]                     # (N, embed_dim)
        scores = np.sum(diff ** 2, axis=1)                         # (N,)

        total_ms = (time.perf_counter() - t0) * 1000
        per_ms = total_ms / len(X)

        return [
            {
                "score": float(scores[i]),
                "is_anomaly": bool(scores[i] > self.delta),
                "embedding": Z[i],
                "latency_ms": per_ms,
            }
            for i in range(len(X))
        ]

    def update_mu_norm(self, new_mu: np.ndarray) -> None:
        """Cập nhật tâm cụm normal (dùng trong incremental learning)."""
        assert new_mu.shape == self.mu_norm.shape, (
            f"Shape mismatch: {new_mu.shape} vs {self.mu_norm.shape}"
        )
        self.mu_norm = new_mu.copy()

    def update_delta(self, new_delta: float) -> None:
        """Cập nhật ngưỡng anomaly."""
        self.delta = float(new_delta)
        logger.info(f"Delta updated: {self.delta:.4f}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _validate_dir(self):
        required = ["config.json", "mu_norm.npy", "delta.npy"]
        missing = [f for f in required if not (self.model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing files in {self.model_dir}: {missing}"
            )

    def _init_backend(self, backend: str):
        if backend == "fp32":
            return _TorchBackend(self.model_dir, use_int8=False)
        elif backend == "int8":
            return _TorchBackend(self.model_dir, use_int8=True)
        elif backend == "onnx":
            try:
                return _OnnxBackend(self.model_dir)
            except ImportError:
                logger.warning("onnxruntime not installed, falling back to fp32")
                return _TorchBackend(self.model_dir, use_int8=False)
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose: {self.BACKENDS}")

    def _scale(self, X: np.ndarray) -> np.ndarray:
        """Scale feature nếu có scaler."""
        if self.scaler is not None:
            return self.scaler.transform(X).astype(np.float32)
        return X.astype(np.float32)

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
    def benchmark(self, n_runs: int = 500, batch_size: int = 1) -> dict:
        """Đo latency và throughput."""
        rng = np.random.default_rng(42)
        test_X = rng.standard_normal((max(n_runs, batch_size), self.input_dim)) * 0.5

        # Warmup
        for i in range(min(20, n_runs)):
            self.predict(test_X[i])

        # Measure
        times = []
        for i in range(n_runs):
            t0 = time.perf_counter()
            self.predict(test_X[i % len(test_X)])
            times.append((time.perf_counter() - t0) * 1000)

        times = np.array(times)
        return {
            "backend": self.backend.backend_name,
            "n_runs": n_runs,
            "mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "throughput_fps": float(1000 / np.mean(times)),
        }


# =====================================================================
# CLI self-test
# =====================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from configs.paths import get_model_dir

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    MODEL_DIR = str(get_model_dir())

    print("=" * 60)
    print("🧪 ETSSLInference Self-Test")
    print("=" * 60)

    for backend in ["fp32", "onnx"]:
        print(f"\n📦 Backend: {backend.upper()}")
        try:
            engine = ETSSLInference(MODEL_DIR, backend=backend)

            # Single predict
            x = np.random.randn(20).astype(np.float32) * 0.5
            res = engine.predict(x)
            print(f"  score={res['score']:.4f} | anomaly={res['is_anomaly']} | {res['latency_ms']:.3f}ms")

            # Benchmark
            bm = engine.benchmark(n_runs=200)
            print(f"  mean={bm['mean_ms']:.3f}ms | p95={bm['p95_ms']:.3f}ms | {bm['throughput_fps']:.1f} fps")

        except Exception as e:
            print(f"  ⚠️ {e}")

    print("\n✅ Inference module test done")
