"""
shared_state.py — Trạng thái quy trình chia sẻ (Pipeline Shared State).

Lưu trữ trạng thái chia sẻ liên tục để có sự gắn kết giữa hệ thống Capture và giao diện Dashboard.
Sẽ tự động kết xuất ra tập tin logs/pipeline_state.json theo cấu trúc chuẩn.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict

logger = logging.getLogger(__name__)


class PipelineStateWriter:
    """
    Bộ đệm (Buffer) lưu trữ các chỉ số mới nhất của quy trình.
    Có khả năng tự động ghi dữ liệu (flush) ra tệp JSON giúp hiển thị liên tục (Live Mode).
    """

    def __init__(
        self,
        state_path: str | Path = "logs/pipeline_state.json",
        maxlen: int = 200
    ):
        """
        Khởi tạo Trình quản lý trạng thái chia sẻ.

        Args:
            state_path: Đường dẫn lưu tệp trạng thái dưới dạng JSON.
            maxlen: Độ dài tối đa cho mỗi mảng đệm.
        """
        self.state_path = Path(state_path)
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Không thể tạo thư mục lưu trạng thái hệ thống tại {self.state_path.parent}: {e}") from e

        self._lock = Lock()
        self._scores: deque[float] = deque(maxlen=maxlen)
        self._is_anomaly: deque[bool] = deque(maxlen=maxlen)
        self._timestamps: deque[str] = deque(maxlen=maxlen)
        self.total_flows: int = 0
        self.total_anomalies: int = 0
        self.mu_updates: int = 0
        self.last_mu_drift: float = 0.0
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
        """
        Ghi nhận một luồng dữ liệu mạng vừa phân tích xong.

        Args:
            score: Điểm phân loại tính toán.
            is_anomaly: Có phải bất thường hay không.
            src: Thông tin nguồn kết nối.
            dst: Thông tin đích kết nối.
            latency_ms: Thời gian suy luận tính bằng giây quy đổi ra ms.
        """
        with self._lock:
            self.total_flows += 1
            if is_anomaly:
                self.total_anomalies += 1
            self._scores.append(score)
            self._is_anomaly.append(is_anomaly)
            # Giữ khung thời gian nhẹ hơn để không làm quá tải tệp json
            self._timestamps.append(datetime.now().strftime("%H:%M:%S.%f")[:-3])
            self._flush()

    def record_mu_update(self, drift: float, update_id: int) -> None:
        """
        Ghi nhận lần cập nhật học gia tăng gần nhất.

        Args:
            drift: Độ dịch chuyển (L2 Norm) so với cấu trúc cũ.
            update_id: ID đánh dấu lần cập nhật.
        """
        with self._lock:
            self.mu_updates = update_id
            self.last_mu_drift = drift
            self.mu_drift_history.append(drift)
            self._flush()

    def _flush(self) -> None:
        """
        Thực hiện ép lưu toàn bộ trạng thái vào ổ đĩa.
        """
        payload: Dict[str, Any] = {
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
        try:
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except OSError as e:
            logger.error("Sự cố ghi file pipeline state (%s): %s", self.state_path, e)


def load_pipeline_state(state_path: str | Path = "logs/pipeline_state.json") -> Dict[str, Any]:
    """
    Hàm tĩnh hỗ trợ việc đọc tệp trạng thái. 
    Tránh lỗi (Safe-load) nếu tệp gặp lỗi cấu trúc hoặc chưa tồn tại.

    Args:
        state_path: Nơi ghi tệp.

    Returns:
        Từ điển mô tả trạng thái, trả về rỗng nếu gặp bất kỳ lỗi nào.
    """
    path = Path(state_path)
    if not path.exists():
        return {}
    try:
        content = path.read_text(encoding="utf-8")
        return json.loads(content)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Không thể đọc trạng thái hiện hành từ ổ lưu trữ: %s", e)
        return {}
