#!/usr/bin/env python3
"""
recalibrate_threshold.py — Script to recalibrate mu_norm and delta using clean local network traffic.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
import numpy as np
from scapy.all import IP, TCP, UDP, sniff, conf

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.paths import get_model_dir, load_config
from pipeline.flow_aggregator import FlowAggregator
from model.inference import ETSSLInference

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("recalibrate")

class CalibrationCollector:
    def __init__(self, engine: ETSSLInference):
        self.engine = engine
        cfg = load_config()["pipeline"]
        self.aggregator = FlowAggregator(
            flow_timeout=cfg["flow_timeout_sec"],
            time_window_sec=cfg.get("time_window_sec"),
            on_flow_ready=self._on_flow_ready,
        )
        self.feature_vectors = []
        self.flow_count = 0

    def _on_flow_ready(self, flow, features):
        self.feature_vectors.append(features)
        self.flow_count += 1
        if self.flow_count % 10 == 0:
            logger.info(f"Collected {self.flow_count} flows...")

    def packet_handler(self, pkt):
        if IP not in pkt or (TCP not in pkt and UDP not in pkt):
            return

        protocol = "TCP" if TCP in pkt else "UDP"
        src_ip, dst_ip = pkt[IP].src, pkt[IP].dst
        src_port = pkt[TCP].sport if TCP in pkt else pkt[UDP].sport
        dst_port = pkt[TCP].dport if TCP in pkt else pkt[UDP].dport
        timestamp = float(pkt.time)

        header_len, payload_len, flags, win_bytes = 0, 0, "", 0
        if TCP in pkt:
            header_len = pkt[TCP].dataofs * 4
            payload_len = len(pkt[TCP].payload)
            flags = str(pkt[TCP].flags)
            win_bytes = pkt[TCP].window
        else:
            header_len = 8
            payload_len = len(pkt[UDP].payload)

        self.aggregator.add_packet(
            src_ip, dst_ip, src_port, dst_port, protocol,
            timestamp, header_len, payload_len, flags, win_bytes,
        )

def main():
    parser = argparse.ArgumentParser(description="Recalibrate ET-SSL Anomaly Detection Threshold")
    parser.add_argument("--iface", type=str, help="Network interface for live capture (default: Scapy default)")
    parser.add_argument("--pcap", type=str, help="Path to PCAP file for offline calibration")
    parser.add_argument("--duration", type=int, default=600, help="Duration of live capture in seconds (default: 600s / 10m)")
    parser.add_argument("--percentile", type=float, default=99.0, help="Percentile for threshold delta (default: 99.0)")
    parser.add_argument("--kappa", type=float, default=1.0, help="Sensitivity multiplier (default: 1.0)")
    args = parser.parse_args()

    model_dir = get_model_dir()
    logger.info(f"Loading ETSSLInference with model_dir={model_dir}")
    engine = ETSSLInference(str(model_dir), backend="onnx")

    collector = CalibrationCollector(engine)

    if args.pcap:
        logger.info(f"Reading from offline PCAP: {args.pcap}")
        sniff(offline=args.pcap, prn=collector.packet_handler, store=0)
        collector.aggregator.flush_all()
    else:
        iface = args.iface or conf.iface
        logger.info(f"Starting live capture on interface '{iface}' for {args.duration}s...")
        logger.info("Press Ctrl+C to stop early and calculate threshold immediately.")
        try:
            sniff(iface=iface, prn=collector.packet_handler, timeout=args.duration, store=0)
        except KeyboardInterrupt:
            logger.info("Capture stopped by user.")
        finally:
            collector.aggregator.flush_all()

    n_samples = len(collector.feature_vectors)
    if n_samples == 0:
        logger.error("No flows were captured/extracted. Cannot recalibrate threshold.")
        sys.exit(1)

    logger.info(f"Total clean flows collected: {n_samples}")
    X = np.vstack(collector.feature_vectors)

    # 1. Scale features using the fitted StandardScaler
    logger.info("Scaling features...")
    X_scaled = engine._scale(X)

    # 2. Extract embeddings
    logger.info("Running encoder to extract embeddings...")
    z = engine.backend.predict_batch(X_scaled)

    # 3. Calculate new mu_norm (mean embedding of clean flows)
    new_mu_norm = np.mean(z, axis=0)

    # 4. Calculate reconstruction scores using the new mu_norm
    scores = np.sum((z - new_mu_norm) ** 2, axis=1)

    # 5. Propose new delta threshold based on percentile
    new_delta = np.percentile(scores, args.percentile)
    effective_delta = new_delta * args.kappa

    logger.info("=== CALIBRATION STATISTICS ===")
    logger.info(f"Min Score: {np.min(scores):.4f}")
    logger.info(f"Max Score: {np.max(scores):.4f}")
    logger.info(f"Mean Score: {np.mean(scores):.4f}")
    logger.info(f"Std Score: {np.std(scores):.4f}")
    logger.info(f"Proposed Delta ({args.percentile}th percentile): {new_delta:.4f}")
    logger.info(f"Effective Delta (Delta * kappa={args.kappa}): {effective_delta:.4f}")

    # 6. Save the new parameters
    logger.info("Saving new calibration parameters...")
    
    # Save mu_norm.npy
    mu_path = model_dir / "mu_norm.npy"
    np.save(mu_path, new_mu_norm.astype(np.float32))
    logger.info(f"Saved new mu_norm to {mu_path}")

    # Save delta.npy
    delta_path = model_dir / "delta.npy"
    np.save(delta_path, np.array([new_delta], dtype=np.float32))
    logger.info(f"Saved new delta to {delta_path}")

    # Update config.json
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)
        model_config["delta"] = float(new_delta)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(model_config, f, indent=2)
        logger.info(f"Updated delta in {config_path}")

    logger.info("Threshold recalibration successfully completed!")

if __name__ == "__main__":
    main()
