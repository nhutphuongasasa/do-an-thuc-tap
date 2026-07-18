"""Central path helpers — đọc model_dir từ configs/config.yaml."""

from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_dir() -> Path:
    """Thư mục chứa encoder, mu_norm, delta, scaler, ..."""
    model_dir = load_config()["model"]["model_dir"]
    path = Path(model_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()
