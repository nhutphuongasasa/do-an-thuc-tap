"""
alert_manager.py — Ghi log alert ra file (không cần dashboard hay webhook).

2 file output:
  logs/alerts.jsonl        — mỗi dòng 1 event JSON (append, thuận tiện stream/grep)
  logs/alerts_summary.json — snapshot tổng hợp cập nhật sau mỗi event
                              (load thẳng bằng json.load để làm dashboard sau này)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class AlertManager:
    """Ghi alert + normal flow ra JSONL và summary JSON."""

    def __init__(
        self,
        log_path: str | Path = "logs/alerts.jsonl",
        summary_path: str | Path | None = None,
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Summary JSON nằm cùng thư mục với JSONL nếu không chỉ rõ
        self.summary_path = Path(summary_path) if summary_path else (
            self.log_path.parent / "alerts_summary.json"
        )

        self._lock = Lock()
        self._total_alerts = 0
        self._total_normal = 0
        self._recent_alerts: list[dict] = []   # giữ tối đa 100 alert gần nhất trong RAM
        self._MAX_RECENT = 100

        # Khởi tạo summary nếu chưa có
        if not self.summary_path.exists():
            self._flush_summary()

    # ──────────────────────────────────────────────────────────────────────────
    def raise_alert(
        self,
        *,
        score: float,
        delta: float,
        flow_id: str,
        src: str,
        dst: str,
        extra: dict[str, Any] | None = None,
    ) -> dict:
        """Ghi alert khi flow vượt ngưỡng δ — ra JSONL + summary."""
        record: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "type": "anomaly",
            "score": round(score, 4),
            "delta": round(delta, 4),
            "flow_id": flow_id,
            "src": src,
            "dst": dst,
        }
        if extra:
            record.update(extra)

        with self._lock:
            self._append_jsonl(record)
            self._total_alerts += 1
            self._recent_alerts.append(record)
            if len(self._recent_alerts) > self._MAX_RECENT:
                self._recent_alerts.pop(0)
            self._flush_summary()

        logger.warning(
            "[ANOMALY] %s | score=%.2f > δ=%.2f", flow_id, score, delta
        )
        return record

    # ──────────────────────────────────────────────────────────────────────────
    def log_normal(self, *, score: float, flow_id: str, src: str, dst: str) -> None:
        """Ghi flow bình thường vào summary (không ra JSONL để tránh spam file)."""
        with self._lock:
            self._total_normal += 1
            # Chỉ cập nhật summary mỗi 50 flow thường — giảm write I/O
            if self._total_normal % 50 == 0:
                self._flush_summary()

    # ──────────────────────────────────────────────────────────────────────────
    @property
    def total_alerts(self) -> int:
        return self._total_alerts

    # ─── Nội bộ ───────────────────────────────────────────────────────────────────
    def _append_jsonl(self, record: dict) -> None:
        """Append 1 dòng JSON vào file .jsonl."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _flush_summary(self) -> None:
        """
        Ghi snapshot tổng hợp ra alerts_summary.json.
        File này luôn chứa state mới nhất — load thẳng bằng json.load() để dùng sau.
        """
        summary = {
            "last_updated": datetime.now().isoformat(),
            "total_alerts": self._total_alerts,
            "total_normal_flows": self._total_normal,
            "total_flows": self._total_alerts + self._total_normal,
            "alert_rate_pct": round(
                100 * self._total_alerts / max(1, self._total_alerts + self._total_normal), 2
            ),
            "recent_alerts": self._recent_alerts[-self._MAX_RECENT:],
            "jsonl_log": str(self.log_path),
        }
        tmp = self.summary_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        tmp.replace(self.summary_path)  # atomic replace — tránh đọc file rỗng khi đang ghi
