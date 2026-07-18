"""
inference.py — Cỗ máy suy luận ET-SSL (Inference Engine).

Hàm suy luận thuần túy: S(x) = ||f_θ(x) - μ_norm||²
Hỗ trợ 3 backend: PyTorch FP32, PyTorch INT8, ONNX Runtime (chỉ dùng CPU).
Mô-đun này độc lập hoàn toàn với mã huấn luyện.

Tham chiếu: Sattar et al. 2025, mục "Anomaly Detection"
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol

import joblib
import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
# Khai báo Giao thức (Strategy Pattern)
# =====================================================================
class InferenceBackend(Protocol):
    """
    Giao thức chuẩn cho các mô-đun thực thi suy luận (Backend).
    Mọi backend (PyTorch, ONNX) đều phải tuân thủ chữ ký hàm này.
    """
    backend_name: str

    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        """
        Thực hiện dự đoán cho một lô dữ liệu.

        Args:
            x: Mảng numpy chứa vector đặc trưng đầu vào, shape (N, input_dim).

        Returns:
            Mảng numpy chứa vector nhúng đầu ra, shape (N, embed_dim).
        """
        ...


# =====================================================================
# Backend: PyTorch
# =====================================================================
class _TorchBackend:
    """Backend thực thi suy luận sử dụng PyTorch (FP32 hoặc INT8)."""

    def __init__(self, model_dir: Path, use_int8: bool = False):
        import torch
        from model.encoder import load_encoder

        fname = "encoder_int8.pt" if use_int8 else "encoder_fp32.pt"
        weights_path = model_dir / fname
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp trọng số PyTorch tại: {weights_path}")

        try:
            self.model = load_encoder(
                weights_path=weights_path,
                config_path=model_dir / "config.json",
                device="cpu",
            )
        except Exception as e:
            raise RuntimeError(f"Lỗi khi nạp mô hình PyTorch: {e}") from e
            
        self._torch = torch
        self.backend_name = "PyTorch-INT8" if use_int8 else "PyTorch-FP32"

    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        tensor = self._torch.from_numpy(x.astype(np.float32))
        with self._torch.no_grad():
            z = self.model(tensor).numpy()
        return z


# =====================================================================
# Backend: ONNX Runtime
# =====================================================================
class _OnnxBackend:
    """Backend thực thi suy luận sử dụng ONNX Runtime, tối ưu cho CPU thiết bị biên."""

    def __init__(self, model_dir: Path):
        import onnxruntime as ort

        onnx_path = model_dir / "encoder_v5.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp mô hình ONNX tại: {onnx_path}")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Giới hạn 1 luồng để mô phỏng và đo lường ổn định trên thiết bị biên
        opts.intra_op_num_threads = 1  

        try:
            self.session = ort.InferenceSession(
                str(onnx_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        except Exception as e:
            raise RuntimeError(f"Lỗi khởi tạo phiên làm việc ONNX: {e}") from e
            
        self.input_name = self.session.get_inputs()[0].name
        self.backend_name = "ONNX-CPU"

    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x.astype(np.float32)})[0]


# =====================================================================
# Cỗ máy suy luận chính (ET-SSL Inference Engine)
# =====================================================================
class ETSSLInference:
    """
    Cỗ máy suy luận cho bài toán phát hiện bất thường ET-SSL.

    Công thức điểm số: S(x) = ||f_θ(x_scaled) - μ_norm||²
    Điều kiện phân loại: Bất thường nếu S(x) > δ
    """

    BACKENDS = ("fp32", "int8", "onnx")

    def __init__(
        self,
        model_dir: str | Path,
        backend: Literal["fp32", "int8", "onnx"] = "onnx",
        delta_override: Optional[float] = None,
    ):
        """
        Khởi tạo cỗ máy suy luận.

        Args:
            model_dir: Đường dẫn thư mục chứa cấu trúc mô hình.
            backend: Tên backend suy luận.
            delta_override: Ghi đè giá trị ngưỡng phát hiện (nếu có).
        """
        self.model_dir = Path(model_dir)
        self._validate_dir()

        # Nạp cấu hình kiến trúc
        config_path = self.model_dir / "config.json"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Tệp config.json bị lỗi định dạng: {e}") from e

        self.input_dim = self.config.get("input_dim")
        self.embed_dim = self.config.get("embed_dim")
        if not isinstance(self.input_dim, int) or self.input_dim <= 0:
            raise ValueError(f"Tham số input_dim không hợp lệ: {self.input_dim}")
        if not isinstance(self.embed_dim, int) or self.embed_dim <= 0:
            raise ValueError(f"Tham số embed_dim không hợp lệ: {self.embed_dim}")

        # Nạp vector tâm cụm bình thường
        try:
            self.mu_norm = np.load(self.model_dir / "mu_norm.npy")
        except Exception as e:
            raise RuntimeError(f"Lỗi khi đọc mu_norm.npy: {e}") from e

        # Nạp ngưỡng phát hiện
        try:
            delta_raw = np.load(self.model_dir / "delta.npy")
        except Exception as e:
            raise RuntimeError(f"Lỗi khi đọc delta.npy: {e}") from e
            
        self.delta = float(delta_override or delta_raw.flat[0])

        # Nạp công cụ chuẩn hóa dữ liệu
        scaler_path = self.model_dir / "scaler.pkl"
        if scaler_path.exists():
            try:
                self.scaler = joblib.load(scaler_path)
            except Exception as e:
                raise RuntimeError(f"Lỗi nạp tệp scaler.pkl: {e}") from e
        else:
            self.scaler = None
            logger.warning(
                "Không tìm thấy scaler.pkl. Đặc trưng sẽ được dùng trực tiếp. "
                "Cảnh báo: Hãy đảm bảo dữ liệu đầu vào đã được chuẩn hóa!"
            )

        self.backend: InferenceBackend = self._init_backend(backend)
        logger.info(
            "Cỗ máy ETSSLInference sẵn sàng | backend=%s | δ=%.4f | input_dim=%d",
            self.backend.backend_name, self.delta, self.input_dim
        )

    # ------------------------------------------------------------------
    # Giao tiếp Công khai
    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Tính toán điểm số bất thường cho một vector mẫu.

        Args:
            x: Vector đặc trưng nguyên bản, shape (input_dim,).

        Returns:
            Từ điển kết quả bao gồm: score, is_anomaly, embedding, latency_ms.
        """
        t0 = time.perf_counter()

        x_scaled = self._scale(x.reshape(1, -1))
        z = self.backend.predict_batch(x_scaled)
        score = float(np.sum((z[0] - self.mu_norm) ** 2))

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "score": score,
            "is_anomaly": score > self.delta,
            "embedding": z[0],
            "latency_ms": latency_ms,
        }

    def predict_batch(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """
        Tính toán cho nhiều vector song song.

        Args:
            X: Mảng đặc trưng 2D, shape (N, input_dim).

        Returns:
            Danh sách từ điển kết quả.
        """
        t0 = time.perf_counter()

        X_scaled = self._scale(X)
        Z = self.backend.predict_batch(X_scaled)
        diff = Z - self.mu_norm[np.newaxis, :]
        scores = np.sum(diff ** 2, axis=1)

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
        """Cập nhật tâm cụm bình thường từ bộ học gia tăng."""
        if new_mu.shape != self.mu_norm.shape:
            raise ValueError(f"Khác biệt chiều: {new_mu.shape} so với {self.mu_norm.shape}")
        self.mu_norm = new_mu.copy()

    def update_delta(self, new_delta: float) -> None:
        """Cập nhật linh hoạt ngưỡng phát hiện."""
        self.delta = float(new_delta)
        logger.info("Đã cập nhật ngưỡng Delta: %.4f", self.delta)

    # ------------------------------------------------------------------
    # Công cụ Nội bộ
    # ------------------------------------------------------------------
    def _validate_dir(self) -> None:
        """Kiểm tra sự tồn tại của tập tin cốt lõi."""
        required = ["config.json", "mu_norm.npy", "delta.npy"]
        missing = [f for f in required if not (self.model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"Thư mục mô hình thiếu các tệp thiết yếu: {missing}")

    def _init_backend(self, backend: str) -> InferenceBackend:
        """Khởi tạo mô-đun thực thi theo Strategy Pattern."""
        if backend == "fp32":
            return _TorchBackend(self.model_dir, use_int8=False)
        elif backend == "int8":
            return _TorchBackend(self.model_dir, use_int8=True)
        elif backend == "onnx":
            try:
                import onnxruntime  # noqa
                return _OnnxBackend(self.model_dir)
            except ImportError:
                logger.warning("Thư viện onnxruntime chưa được cài đặt. Hệ thống dự phòng sang fp32.")
                return _TorchBackend(self.model_dir, use_int8=False)
        else:
            raise ValueError(f"Backend không được hỗ trợ: {backend}. Các lựa chọn khả dụng: {self.BACKENDS}")

    def _scale(self, X: np.ndarray) -> np.ndarray:
        """Thực thi chuẩn hóa đặc trưng."""
        if self.scaler is not None:
            return self.scaler.transform(X).astype(np.float32)
        return X.astype(np.float32)

    # ------------------------------------------------------------------
    # Đo lường Hiệu năng
    # ------------------------------------------------------------------
    def benchmark(self, n_runs: int = 500, batch_size: int = 1) -> Dict[str, Any]:
        """Đo lường độ trễ (latency) và băng thông (throughput) của engine."""
        rng = np.random.default_rng(42)
        test_X = rng.standard_normal((max(n_runs, batch_size), self.input_dim)) * 0.5

        self._run_warmup(test_X, min(20, n_runs))
        return self._measure_latency(test_X, n_runs)

    def _run_warmup(self, test_X: np.ndarray, n_warmup: int) -> None:
        """Khởi động bộ nhớ đệm và tối ưu trình biên dịch."""
        for i in range(n_warmup):
            self.predict(test_X[i])

    def _measure_latency(self, test_X: np.ndarray, n_runs: int) -> Dict[str, Any]:
        """Tiến hành vòng lặp đo lường tốc độ xử lý."""
        times = []
        for i in range(n_runs):
            t0 = time.perf_counter()
            self.predict(test_X[i % len(test_X)])
            times.append((time.perf_counter() - t0) * 1000)

        times_arr = np.array(times)
        return {
            "backend": self.backend.backend_name,
            "n_runs": n_runs,
            "mean_ms": float(np.mean(times_arr)),
            "median_ms": float(np.median(times_arr)),
            "p95_ms": float(np.percentile(times_arr, 95)),
            "p99_ms": float(np.percentile(times_arr, 99)),
            "throughput_fps": float(1000 / np.mean(times_arr)),
        }


# =====================================================================
# Mã Tự Kiểm Tra (CLI Self-Test)
# =====================================================================
if __name__ == "__main__":
    import sys
    from configs.paths import get_model_dir

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model_dir = get_model_dir()
    logger.info("🧪 Khởi động bài tự kiểm tra ETSSLInference...")

    for be in ["fp32", "onnx"]:
        logger.info("📦 Thử nghiệm Backend: %s", be.upper())
        try:
            engine = ETSSLInference(model_dir, backend=be)
            
            x_test = np.random.randn(20).astype(np.float32) * 0.5
            res = engine.predict(x_test)
            logger.info(
                "  Dự đoán mẫu: score=%.4f | anomaly=%s | trễ=%.3fms",
                res["score"], res["is_anomaly"], res["latency_ms"]
            )

            bm = engine.benchmark(n_runs=200)
            logger.info(
                "  Hiệu suất: trung bình=%.3fms | p95=%.3fms | băng thông=%.1f fps",
                bm["mean_ms"], bm["p95_ms"], bm["throughput_fps"]
            )

        except Exception as err:
            logger.error("⚠️ Sự cố xảy ra với backend %s: %s", be, err)

    logger.info("✅ Hoàn tất kiểm tra mô-đun suy luận.")
