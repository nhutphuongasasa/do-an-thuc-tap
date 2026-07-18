"""
capture.py — Packet Capture (Scapy) → Flow → Feature → Inference → Alert → Incremental.

Flow khớp readme §4 sequence diagram:
  NIC/pcap → Capture → Flow Aggregator → Feature Extractor → Inference → Alert → μ_norm update
"""

import argparse
import logging
import sys
from pathlib import Path

from scapy.all import IP, TCP, UDP, sniff

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.paths import get_model_dir, load_config
from pipeline.flow_aggregator import FlowAggregator
from pipeline.inference_runner import PipelineInferenceRunner

logger = logging.getLogger(__name__)


class TrafficCapture:
    """Bắt gói tin live hoặc replay pcap, chạy full ET-SSL pipeline."""

    def __init__(self, runner: PipelineInferenceRunner):
        self.runner = runner
        cfg = load_config()["pipeline"]
        self.aggregator = FlowAggregator(
            flow_timeout=cfg["flow_timeout_sec"],
            time_window_sec=cfg.get("time_window_sec"),
            on_flow_ready=self._on_flow_ready,
        )

    def _on_flow_ready(self, flow, features):
        self.runner.process_flow(flow, features)

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

        forward_key = (src_ip, dst_ip, src_port, dst_port, protocol)
        backward_key = (dst_ip, src_ip, dst_port, src_port, protocol)
        _ = forward_key, backward_key  # aggregator resolves direction internally

        self.aggregator.add_packet(
            src_ip, dst_ip, src_port, dst_port, protocol,
            timestamp, header_len, payload_len, flags, win_bytes,
        )

    def run_live(self, interface=None):
        logger.info("Live capture on interface: %s", interface or "default")
        sniff(iface=interface, prn=self.packet_handler, store=0)

    def run_pcap(self, pcap_file: str):
        logger.info("Reading PCAP: %s", pcap_file)
        sniff(offline=pcap_file, prn=self.packet_handler, store=0)
        self.aggregator.flush_all()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="ET-SSL Traffic Capture Pipeline")
    parser.add_argument("--pcap", type=str, help="Path to PCAP file")
    parser.add_argument("--iface", type=str, help="Network interface for live capture")
    parser.add_argument("--backend", default="onnx", choices=["onnx", "fp32", "int8"])
    args = parser.parse_args()

    model_dir = get_model_dir()
    logger.info("Loading pipeline (model=%s, backend=%s)...", model_dir, args.backend)
    runner = PipelineInferenceRunner.from_config(model_dir, backend=args.backend)
    capture = TrafficCapture(runner)

    if args.pcap:
        capture.run_pcap(args.pcap)
    else:
        capture.run_live(args.iface)


if __name__ == "__main__":
    main()
