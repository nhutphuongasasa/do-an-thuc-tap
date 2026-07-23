"""
logger.py — Ghi log JSONL, JSON summary, và in terminal.

3 nhiệm vụ tách biệt:
  1. log_flow(): ghi từng flow vào logs/flow_decisions.jsonl
  2. print_stats(): in 1 dòng tổng hợp mỗi giây ra terminal
  3. write_summary(): ghi đè logs/stats_summary.json mỗi 60s

Chạy trong thread riêng, đọc từ result_queue.
"""

import json
import time
import logging
import sys
from pathlib import Path
from queue import Queue, Empty

log = logging.getLogger("logger")

FLOW_LOG_PATH    = Path("logs/flow_decisions.jsonl")
SUMMARY_LOG_PATH = Path("logs/stats_summary.json")

SUMMARY_INTERVAL_S  = 60.0
TERMINAL_INTERVAL_S = 1.0

# Số đặc trưng có giá trị cao nhất để ghi vào log (top_features)
TOP_N_FEATURES = 5


def _top_features(feat_vec, feature_names):
    """Trả về dict N đặc trưng có giá trị tuyệt đối lớn nhất."""
    indexed = sorted(enumerate(feat_vec), key=lambda x: abs(x[1]), reverse=True)
    return {feature_names[i]: float(v) for i, v in indexed[:TOP_N_FEATURES]}


def run(result_queue: Queue, stats: dict, stop_event, feature_names: list):
    """
    Vòng lặp chính của logger.
    stats là dict chia sẻ chung với các module khác (đọc để tổng hợp).
    """
    FLOW_LOG_PATH.parent.mkdir(exist_ok=True)
    flow_log = FLOW_LOG_PATH.open("a", encoding="utf-8", buffering=1)  # line-buffered

    anomaly_count    = 0
    flow_count       = 0
    last_terminal_ts = time.monotonic()
    last_summary_ts  = time.monotonic()

    # Snapshot để tính rate theo giây
    last_pkt_total   = 0
    last_flow_total  = 0

    while not stop_event.is_set() or not result_queue.empty():
        # Xử lý tất cả kết quả sẵn có trong queue
        while True:
            try:
                res = result_queue.get_nowait()
            except Empty:
                break

            flow_count += 1
            flow_data = res["flow_data"]
            pkts      = flow_data["packets"]
            duration  = (pkts[-1][0] - pkts[0][0]) if len(pkts) > 1 else 0.0

            # Ghi JSONL
            record = {
                "ts":              time.time(),
                "flow_id":         res["flow_id"],
                "score":           round(res["score"], 6),
                "delta_effective": round(res["delta"], 6),
                "is_anomaly":      res["is_anomaly"],
                "packet_count":    len(pkts),
                "duration_s":      round(duration, 4),
                "top_features":    _top_features(res["feature_vector"], feature_names),
            }
            flow_log.write(json.dumps(record) + "\n")

            # In ngay khi có anomaly — không cần chờ đến kỳ terminal
            if res["is_anomaly"]:
                anomaly_count += 1
                print(
                    f"\033[91m[ANOMALY]\033[0m {res['flow_id']} "
                    f"score={res['score']:.4f} Δ={res['delta']:.4f}",
                    flush=True,
                )

        now = time.monotonic()

        # In 1 dòng tổng hợp mỗi giây
        if now - last_terminal_ts >= TERMINAL_INTERVAL_S:
            elapsed  = now - last_terminal_ts
            pkt_now  = stats.get("pkt_total", 0)
            flow_now = flow_count

            pkt_rate  = (pkt_now  - last_pkt_total)  / elapsed
            flow_rate = (flow_now - last_flow_total)  / elapsed

            last_pkt_total   = pkt_now
            last_flow_total  = flow_now
            last_terminal_ts = now

            anom_rate = anomaly_count / max(flow_count, 1)
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"pkts/s={pkt_rate:.0f}  flows/s={flow_rate:.1f}  "
                f"anomalies={anomaly_count}({anom_rate:.1%})  "
                f"queue_drops={stats.get('queue_drops', 0)}",
                flush=True,
            )

        # Ghi summary JSON mỗi 60s
        if now - last_summary_ts >= SUMMARY_INTERVAL_S:
            last_summary_ts = now
            total_batches = stats.get("total_batches", 1)
            total_feats   = stats.get("total_batch_feats", 0)
            summary = {
                "ts":                   time.strftime("%Y-%m-%dT%H:%M:%S"),
                "packets_per_sec_avg":  round(stats.get("pkt_total", 0) / max(now, 1), 2),
                "flows_per_sec_avg":    round(flow_count / max(now, 1), 2),
                "anomaly_count":        anomaly_count,
                "anomaly_rate":         round(anomaly_count / max(flow_count, 1), 4),
                "avg_batch_size":       round(total_feats / max(total_batches, 1), 2),
                "avg_latency_ms":       round(stats.get("last_latency_ms", 0), 3),
                "queue_drops":          stats.get("queue_drops", 0),
                "flow_active":          stats.get("flow_active", 0),
            }
            SUMMARY_LOG_PATH.write_text(json.dumps(summary, indent=2))

        time.sleep(0.05)  # nhường CPU, không spin-loop

    flow_log.close()
    log.info("Logger đã dừng.")
