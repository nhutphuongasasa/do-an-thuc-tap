"""
alert_manager.py — Hệ thống lưu trữ cảnh báo (Alert Manager).

Ghi log các sự kiện cảnh báo ra tệp tin (không yêu cầu webhook hoặc dashboard).
Bao gồm 2 tệp đầu ra chính:
  logs/alerts.jsonl        — Nhật ký dạng JSON Lines, mỗi sự kiện một dòng (thuận tiện cho stream/grep).
  logs/alerts_summary.json — Bản tóm tắt trạng thái (snapshot), cập nhật liên tục sau mỗi sự kiện
                               (dễ dàng đọc bằng json.load() để tích hợp làm giao diện điều khiển sau này).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Trình quản lý hệ thống ghi nhật ký: Xử lý việc ghi luồng dữ liệu (bình thường và bất thường)
    vào tệp JSONL và tệp tóm tắt JSON một cách an toàn đa luồng (thread-safe).
    """

    def __init__(
        self,
        log_path: str | Path = "logs/alerts.jsonl",
        summary_path: Optional[str | Path] = None,
    ):
        """
        Khởi tạo Trình quản lý cảnh báo.

        Args:
            log_path: Đường dẫn tệp nhật ký cảnh báo JSONL.
            summary_path: Đường dẫn tệp tóm tắt JSON. Nếu None, tệp sẽ được tạo tự động cùng thư mục với log_path.
        """
        self.log_path = Path(log_path)
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Không thể tạo thư mục lưu trữ log tại {self.log_path.parent}: {e}") from e

        # Tệp tóm tắt JSON tự động nằm cùng thư mục với JSONL nếu không được chỉ định rõ
        self.summary_path = Path(summary_path) if summary_path else (
            self.log_path.parent / "alerts_summary.json"
        )

        self._lock = Lock()
        self._total_alerts = 0
        self._total_normal = 0
        self._recent_alerts: List[Dict[str, Any]] = []   # Giữ tối đa 100 cảnh báo gần nhất trong bộ nhớ
        self._MAX_RECENT = 100

        # Khởi tạo tệp tóm tắt nếu chưa tồn tại
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
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Kích hoạt cảnh báo khi một luồng vượt ngưỡng giới hạn δ.
        Hệ thống sẽ ghi log ra JSONL và cập nhật file tóm tắt đồng thời.

        Args:
            score: Điểm bất thường (Anomaly Score).
            delta: Ngưỡng giới hạn cho phép (Anomaly Threshold).
            flow_id: Mã định danh luồng (thường là dạng nguồn->đích).
            src: Thông tin nguồn.
            dst: Thông tin đích.
            extra: Từ điển chứa các thông tin phụ trợ bổ sung (tuỳ chọn).

        Returns:
            Từ điển chứa bản ghi dữ liệu sự kiện đã được lưu.
        """
        record: Dict[str, Any] = {
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
        """
        Ghi nhận luồng dữ liệu bình thường vào bộ tóm tắt.
        Không ghi ra tệp JSONL để hạn chế tối đa việc lãng phí I/O (tránh spam tệp log).

        Args:
            score: Điểm bất thường của luồng.
            flow_id: Định danh luồng.
            src: Điểm nguồn.
            dst: Điểm đích.
        """
        with self._lock:
            self._total_normal += 1
            # Tối ưu: Chỉ xả nội dung tóm tắt ra đĩa sau mỗi 50 sự kiện thông thường
            if self._total_normal % 50 == 0:
                self._flush_summary()

    # ──────────────────────────────────────────────────────────────────────────
    @property
    def total_alerts(self) -> int:
        """Tổng số cảnh báo đã được hệ thống kích hoạt."""
        return self._total_alerts

    # ─── Nội bộ ───────────────────────────────────────────────────────────────────
    def _append_jsonl(self, record: Dict[str, Any]) -> None:
        """
        Mở và ghi chèn thêm (append) 1 dòng JSON vào tệp .jsonl.
        """
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error("Không thể ghi log sự kiện cảnh báo ra đĩa (%s): %s", self.log_path, e)

    def _flush_summary(self) -> None:
        """
        Kết xuất ảnh chụp nhanh (snapshot) thống kê ra tệp alerts_summary.json.
        Áp dụng kỹ thuật thay thế nguyên tử (atomic replace) để ngăn việc đọc lỗi khi ghi đang diễn ra.
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
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            tmp.replace(self.summary_path)
        except OSError as e:
            logger.error("Không thể cập nhật tệp thống kê tóm tắt (%s): %s", self.summary_path, e)
