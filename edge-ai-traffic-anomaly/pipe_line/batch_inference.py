"""
batch_inference.py — Gom flow thành micro-batch rồi gọi ONNX một lần duy nhất.

Flush batch khi:
  - Đủ max_batch flow, HOẶC
  - Đã chờ quá max_wait_ms ms (whichever comes first).

Incremental μ_norm update: mỗi 60s, dùng các embedding của flow có score < 0.5×Δ.
"""

import time
import logging
import numpy as np
from queue import Queue, Empty

log = logging.getLogger("batch_inference")

# Tần suất cập nhật μ_norm (giây)
MU_UPDATE_INTERVAL_S = 60.0

# Chỉ dùng flow "rõ ràng bình thường" để update μ_norm
NORMAL_SCORE_RATIO   = 0.5   # score < 0.5 × Δ


def run(flow_queue: Queue, result_queue: Queue, engine, stats: dict, stop_event,
        max_batch: int = 32, max_wait_ms: float = 200.0):
    """
    Đọc flow từ flow_queue, gom batch, gọi engine.predict_batch(), đẩy kết quả
    vào result_queue để logger xử lý.

    engine: ETSSLInference (đã khởi tạo ở main.py).
    """
    max_wait_s     = max_wait_ms / 1000.0
    mu_candidates  = []         # embedding của flow bình thường để update μ_norm
    last_mu_update = time.monotonic()

    pending_ids    = []         # flow_id
    pending_flows  = []         # flow_data dict
    pending_feats  = []         # np.ndarray (20,)
    batch_deadline = None       # monotonic time — khi nào phải flush dù chưa đủ batch

    while not stop_event.is_set() or not flow_queue.empty():
        # Cố lấy thêm flow vào batch
        try:
            flow_id, flow_data, feat = flow_queue.get(timeout=0.02)
            pending_ids.append(flow_id)
            pending_flows.append(flow_data)
            pending_feats.append(feat)

            # Bắt đầu đếm deadline từ flow đầu tiên trong batch
            if batch_deadline is None:
                batch_deadline = time.monotonic() + max_wait_s
        except Empty:
            pass

        now = time.monotonic()
        should_flush = (
            len(pending_feats) >= max_batch
            or (batch_deadline is not None and now >= batch_deadline)
        )

        if should_flush and pending_feats:
            t0 = time.perf_counter()

            X = np.vstack(pending_feats)             # (N, 20) — 1 lần numpy alloc
            results = engine.predict_batch(X)        # list of dict

            latency_ms = (time.perf_counter() - t0) * 1000
            stats["last_batch_size"]   = len(results)
            stats["last_latency_ms"]   = latency_ms
            stats["total_batches"]     = stats.get("total_batches", 0) + 1
            stats["total_batch_feats"] = stats.get("total_batch_feats", 0) + len(results)

            delta = engine.effective_delta

            for i, res in enumerate(results):
                score      = float(res["score"])
                is_anomaly = bool(res["is_anomaly"])
                embedding  = res["embedding"]

                result_queue.put_nowait({
                    "flow_id":        pending_ids[i],
                    "flow_data":      pending_flows[i],
                    "feature_vector": pending_feats[i],
                    "score":          score,
                    "delta":          delta,
                    "is_anomaly":     is_anomaly,
                    "latency_ms":     latency_ms / len(results),
                })

                # Gom embedding bình thường để update μ_norm sau
                if not is_anomaly and score < NORMAL_SCORE_RATIO * delta:
                    mu_candidates.append(embedding)

            # Reset batch
            pending_ids.clear()
            pending_flows.clear()
            pending_feats.clear()
            batch_deadline = None

        # Cập nhật μ_norm định kỳ (không cập nhật từng flow — quá tốn CPU)
        if (now - last_mu_update) >= MU_UPDATE_INTERVAL_S and mu_candidates:
            new_mu = np.mean(mu_candidates, axis=0).astype(np.float32)
            engine.update_mu_norm(new_mu)
            log.info("μ_norm cập nhật từ %d flow bình thường.", len(mu_candidates))
            mu_candidates.clear()
            last_mu_update = now

    log.info("Batch inference đã dừng.")
