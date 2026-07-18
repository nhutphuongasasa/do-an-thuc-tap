"""
demo_stream.py — Demo traffic qua full pipeline (không cần pcap).

Chạy inference + alert + incremental learning, ghi shared state cho dashboard.
Flow khớp readme §0 khi không có traffic thật.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.paths import get_model_dir, load_config
from pipeline.feature_extractor import FlowRecord
from pipeline.inference_runner import PipelineInferenceRunner

logger = logging.getLogger(__name__)


def _make_flow(idx: int, anomaly: bool, rng: np.random.Generator) -> tuple[FlowRecord, np.ndarray]:
    """Tạo flow giả với feature vector."""
    flow = FlowRecord(
        src_ip=f"10.0.0.{idx % 250 + 1}",
        dst_ip="10.0.1.1",
        src_port=40000 + idx,
        dst_port=443,
        protocol="TCP",
    )
    center = 3.0 if anomaly else 0.0
    features = rng.normal(center, 1.2 if anomaly else 0.8, 20).astype(np.float32)
    return flow, features


def run_demo(
    n_flows: int = 500,
    flows_per_sec: float = 5.0,
    anomaly_ratio: float = 0.15,
    backend: str = "onnx",
):
    cfg = load_config()
    model_dir = get_model_dir()
    runner = PipelineInferenceRunner.from_config(model_dir, backend=backend)
    runner.learner.update_interval = cfg["incremental"]["update_interval"]
    runner.learner.alpha = cfg["incremental"]["alpha"]

    rng = np.random.default_rng(42)
    interval = 1.0 / max(flows_per_sec, 0.1)

    logger.info(
        "Demo stream: %d flows @ %.1f fps, anomaly=%.0f%%",
        n_flows, flows_per_sec, anomaly_ratio * 100,
    )

    for i in range(n_flows):
        is_anom = rng.random() < anomaly_ratio
        flow, features = _make_flow(i, is_anom, rng)
        runner.process_flow(flow, features)
        time.sleep(interval)

    if runner.learner:
        runner.learner.force_update()
        runner.engine.update_mu_norm(runner.learner.mu_norm.astype(np.float32))

    logger.info(
        "Done: %d flows, %d alerts, %d μ updates",
        runner._flows_processed,
        runner.alerts.total_alerts,
        runner.learner.stats["total_updates"] if runner.learner else 0,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--flows", type=int, default=0, help="0 = chạy vô hạn")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--anomaly-ratio", type=float, default=0.15)
    parser.add_argument("--backend", default="onnx")
    args = parser.parse_args()

    if args.flows == 0:
        cfg = load_config()
        model_dir = get_model_dir()
        runner = PipelineInferenceRunner.from_config(model_dir, backend=args.backend)
        rng = np.random.default_rng(42)
        interval = 1.0 / max(args.fps, 0.1)
        i = 0
        logger.info("Demo stream vô hạn — Ctrl+C để dừng")
        try:
            while True:
                is_anom = rng.random() < args.anomaly_ratio
                flow, features = _make_flow(i, is_anom, rng)
                runner.process_flow(flow, features)
                i += 1
                time.sleep(interval)
        except KeyboardInterrupt:
            runner.learner.force_update()
            logger.info("Stopped after %d flows", i)
    else:
        run_demo(args.flows, args.fps, args.anomaly_ratio, args.backend)


if __name__ == "__main__":
    main()
