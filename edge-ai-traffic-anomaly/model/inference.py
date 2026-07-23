"""
model/inference.py — ETSSLInference: load encoder ET-SSL đã train + mu_norm + delta,
tính score = ||embedding - mu_norm||^2 (khoảng cách bình phương tới tâm normal).

Cần các file sau trong model_dir:
  - edge_encoder_v4.onnx (+.onnx.data) — encoder ONNX (bắt buộc nếu backend="onnx")
  - encoder_best.pt  — state_dict PyTorch (chỉ cần nếu backend="pkl")
  - config.json      — hidden_dims/embed_dim (chỉ cần nếu backend="pkl")
  - scaler.pkl        — StandardScaler (joblib), BẮT BUỘC cho cả 2 backend
  - mu_norm.npy       — tâm cụm normal, shape (embed_dim,)
  - threshold.json    — {"delta": float, "percentile": 95}

score < 0.5*delta -> coi là "rõ ràng bình thường" (dùng để update mu_norm, xem batch_inference.py)
score > delta     -> is_anomaly = True
"""

import json
import logging
from pathlib import Path

import numpy as np
import joblib

log = logging.getLogger("inference")


class ETSSLInference:
    def __init__(self, model_dir: str, backend: str = "onnx"):
        self.model_dir = Path(model_dir)
        self.backend = backend

        scaler_path = self.model_dir / "scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {scaler_path}. Model ET-SSL train trên feature đã "
                f"StandardScaler — thiếu file này thì predict sẽ sai hoàn toàn."
            )
        self.scaler = joblib.load(scaler_path)

        mu_path = self.model_dir / "mu_norm.npy"
        if not mu_path.exists():
            raise FileNotFoundError(f"Không tìm thấy {mu_path}.")
        self.mu_norm = np.load(mu_path).astype(np.float32)

        th_path = self.model_dir / "threshold.json"
        if not th_path.exists():
            raise FileNotFoundError(f"Không tìm thấy {th_path}.")
        with open(th_path, "r", encoding="utf-8") as f:
            th = json.load(f)
        self.delta = float(th["delta"])

        if backend == "onnx":
            self._load_onnx()
        elif backend == "pkl":
            self._load_torch()
        else:
            raise ValueError(f"backend không hợp lệ: {backend!r} (chỉ nhận 'onnx' hoặc 'pkl')")

        log.info(
            "ETSSLInference sẵn sàng | backend=%s | embed_dim=%d | delta=%.6f",
            backend, self.mu_norm.shape[0], self.delta,
        )

    def _load_onnx(self):
        import onnxruntime as ort

        onnx_files = list(self.model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"Không tìm thấy file .onnx trong {self.model_dir}.")
        self._session = ort.InferenceSession(str(onnx_files[0]), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def _load_torch(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        cfg_path = self.model_dir / "config.json"
        pt_path = self.model_dir / "encoder_best.pt"
        if not cfg_path.exists() or not pt_path.exists():
            raise FileNotFoundError(
                f"Backend 'pkl' cần {cfg_path.name} và {pt_path.name} trong {self.model_dir}."
            )
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        input_dim = len(self.scaler.mean_)

        class Encoder(nn.Module):
            def __init__(self, input_dim, hidden_dims, embed_dim):
                super().__init__()
                layers = []
                prev = input_dim
                for h in hidden_dims:
                    layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
                    prev = h
                layers.append(nn.Linear(prev, embed_dim))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return F.normalize(self.net(x), dim=1)

        self._torch = torch
        self._model = Encoder(input_dim, cfg["hidden_dims"], cfg["embed_dim"])
        self._model.load_state_dict(torch.load(pt_path, map_location="cpu"))
        self._model.eval()

    @property
    def effective_delta(self) -> float:
        return self.delta

    def update_mu_norm(self, new_mu: np.ndarray) -> None:
        self.mu_norm = np.asarray(new_mu, dtype=np.float32)

    def predict_batch(self, X: np.ndarray) -> list[dict]:
        X_scaled = self.scaler.transform(X).astype(np.float32)

        if self.backend == "onnx":
            embeddings = self._session.run([self._output_name], {self._input_name: X_scaled})[0]
        else:
            with self._torch.no_grad():
                embeddings = self._model(self._torch.from_numpy(X_scaled)).numpy()

        scores = np.sum((embeddings - self.mu_norm) ** 2, axis=1)
        is_anomaly = scores > self.delta

        return [
            {
                "score": float(scores[i]),
                "is_anomaly": bool(is_anomaly[i]),
                "embedding": embeddings[i],
            }
            for i in range(len(scores))
        ]
