"""
encoder.py — Định nghĩa kiến trúc Encoder và hàm load model.

Tách biệt hoàn toàn khỏi training code, chỉ dùng cho inference.
Kiến trúc phải giống hệt lúc train (từ config.json: input=20, hidden=256, embed=64).
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
from typing import Optional


class Encoder(nn.Module):
    """
    Encoder f_θ: x_i → z_i
    Kiến trúc MLP 4 layer với BatchNorm và Dropout.
    Tham chiếu: mục "Methodology" trong Sattar et al. 2025.
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 256,
        embed_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            # Layer 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            # Layer 3
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            # Layer 4 (projection head)
            nn.Linear(hidden_dim // 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) — feature vector đã chuẩn hóa
        Returns:
            z: (batch, embed_dim) — embedding
        """
        return self.net(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Alias cho forward — rõ ràng hơn khi dùng trong inference."""
        return self.forward(x)


def load_encoder(
    weights_path: str,
    config_path: Optional[str] = None,
    device: str = "cpu",
) -> Encoder:
    """
    Load encoder từ file .pt.

    Args:
        weights_path: Đường dẫn tới encoder_fp32.pt hoặc encoder_int8.pt
        config_path: Đường dẫn config.json để đọc input_dim, hidden_dim, embed_dim.
                     Nếu None, dùng giá trị mặc định (20, 256, 64).
        device: "cpu" hoặc "cuda" (edge AI dùng cpu)

    Returns:
        Encoder ở eval mode
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    # Đọc kiến trúc từ config nếu có
    input_dim, hidden_dim, embed_dim = 20, 256, 64
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            input_dim = cfg.get("input_dim", input_dim)
            hidden_dim = cfg.get("hidden_dim", hidden_dim)
            embed_dim = cfg.get("embed_dim", embed_dim)

    model = Encoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        embed_dim=embed_dim,
    )

    # Load state dict — xử lý cả trường hợp có prefix 'encoder.'
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    clean_state = {}
    for k, v in state_dict.items():
        new_k = k[len("encoder."):] if k.startswith("encoder.") else k
        clean_state[new_k] = v

    # strict=False để tương thích với các phiên bản có/không có wrapper
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing:
        print(f"[Encoder] Warning: missing keys: {missing}")
    if unexpected:
        print(f"[Encoder] Warning: unexpected keys: {unexpected}")

    model = model.to(device)
    model.eval()
    return model


def get_model_info(model: Encoder) -> dict:
    """Trả về thông tin model: số param, kích thước ước tính."""
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1e6

    return {
        "total_params": n_params,
        "trainable_params": n_trainable,
        "estimated_size_mb": round(size_mb, 3),
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "embed_dim": model.embed_dim,
    }


if __name__ == "__main__":
    # Quick self-test
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from configs.paths import get_model_dir

    MODEL_DIR = get_model_dir()

    print("🔧 Testing Encoder module...")
    enc = load_encoder(
        weights_path=MODEL_DIR / "encoder_fp32.pt",
        config_path=MODEL_DIR / "config.json",
    )
    info = get_model_info(enc)
    print(f"  input_dim={info['input_dim']}, hidden={info['hidden_dim']}, embed={info['embed_dim']}")
    print(f"  Total params: {info['total_params']:,}")
    print(f"  Estimated size: {info['estimated_size_mb']} MB")

    # Test forward pass
    x = torch.randn(1, info["input_dim"])
    with torch.no_grad():
        z = enc(x)
    print(f"  Forward pass: {x.shape} → {z.shape}")
    print("✅ Encoder module OK")
