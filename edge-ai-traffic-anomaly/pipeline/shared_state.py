"""
shared_state.py — Trạng thái pipeline dùng chung (capture ↔ dashboard).

Ghi logs/pipeline_state.json để dashboard đọc ở live mode.
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock


class PipelineStateWriter:
    """Buffer trạng thái gần nhất, flush ra JSON cho dashboard."""

    def __init__(self, state_path: str | Path = "logs/pipeline_state.json", maxlen: int = 200):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._scores: deque[float] = deque(maxlen=maxlen)
        self._is_anomaly: deque[bool] = deque(maxlen=maxlen)
        self._timestamps: deque[str] = deque(maxlen=maxlen)
        self.total_flows = 0
        self.total_anomalies = 0
        self.mu_updates = 0
        self.last_mu_drift = 0.0
        self.mu_drift_history: deque[float] = deque(maxlen=maxlen)

    def record_flow(
        self,
        *,
        score: float,
        is_anomaly: bool,
        src: str,
        dst: str,
        latency_ms: float,
    ) -> None:
        with self._lock:
            self.total_flows += 1
            if is_anomaly:
                self.total_anomalies += 1
            self._scores.append(score)
            self._is_anomaly.append(is_anomaly)
            self._timestamps.append(datetime.now().strftime("%H:%M:%S.%f")[:-3])
            self._flush()

    def record_mu_update(self, drift: float, update_id: int) -> None:
        with self._lock:
            self.mu_updates = update_id
            self.last_mu_drift = drift
            self.mu_drift_history.append(drift)
            self._flush()

    def _flush(self) -> None:
        payload = {
            "updated_at": datetime.now().isoformat(),
            "total_flows": self.total_flows,
            "total_anomalies": self.total_anomalies,
            "mu_updates": self.mu_updates,
            "last_mu_drift": self.last_mu_drift,
            "mu_drift_history": list(self.mu_drift_history),
            "scores": list(self._scores),
            "is_anomaly": list(self._is_anomaly),
            "timestamps": list(self._timestamps),
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def load_pipeline_state(state_path: str | Path = "logs/pipeline_state.json") -> dict:
    path = Path(state_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
