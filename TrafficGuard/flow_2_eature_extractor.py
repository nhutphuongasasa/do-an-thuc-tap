import threading
import time
import queue
import logging
import numpy as np

from .models import FeatureVector

logger = logging.getLogger("nids.features")


def _stats(values):
    """Tra ve (mean, std, max, min), an toan khi list rong -> tra 0.0 het."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype="float64")
    return float(arr.mean()), float(arr.std()), float(arr.max()), float(arr.min())


def _iat(timestamps):
    """Inter-Arrival-Time: khoang cach (giay) giua cac timestamp lien tiep."""
    if len(timestamps) < 2:
        return []
    ts = sorted(timestamps)
    return [ts[i] - ts[i - 1] for i in range(1, len(ts))]


def _active_idle_periods(timestamps, idle_threshold):
    """
    Tai tao logic Active/Idle kieu CICFlowMeter:
    - Sap xep timestamp tang dan.
    - Neu khoang cach giua 2 goi tin lien tiep >= idle_threshold: dong 1 "active period"
      (tu active_start den goi truoc do), mo 1 "idle period" co do dai = khoang cach.
    - Neu < idle_threshold: van con trong active period hien tai (flow dang "ban ron").
    Tra ve (active_durations, idle_durations) - don vi giay.
    """
    ts = sorted(timestamps)
    if len(ts) < 2:
        return [], []

    active_durations = []
    idle_durations = []
    active_start = ts[0]
    last_ts = ts[0]

    for t in ts[1:]:
        gap = t - last_ts
        if gap >= idle_threshold:
            active_durations.append(last_ts - active_start)
            idle_durations.append(gap)
            active_start = t
        last_ts = t

    active_durations.append(last_ts - active_start)
    return active_durations, idle_durations


class FeatureCalculator:
    """Tinh 1 dict feature cho 1 flow trong 1 window - dung chung cho moi WindowFeatureExtractor."""

    def __init__(self, idle_threshold: float):
        self.idle_threshold = idle_threshold

    def compute(self, packets):
        """packets: list[PacketEvent] cua 1 flow trong window. Tra ve dict feature hoac None."""
        if len(packets) < 2:
            return None

        packets = sorted(packets, key=lambda p: p.ts)
        fwd = [p for p in packets if p.direction == "fwd"]
        bwd = [p for p in packets if p.direction == "bwd"]

        all_ts = [p.ts for p in packets]
        duration = max(all_ts) - min(all_ts)
        duration = max(duration, 1e-6)  # tranh chia cho 0 khi cac goi den gan nhu cung luc

        fwd_len = [p.length for p in fwd]
        bwd_len = [p.length for p in bwd]

        fwd_mean, fwd_std, fwd_max, fwd_min = _stats(fwd_len)
        bwd_mean, bwd_std, bwd_max, bwd_min = _stats(bwd_len)

        total_bytes = sum(fwd_len) + sum(bwd_len)
        flow_bytes_per_s = total_bytes / duration
        flow_packets_per_s = len(packets) / duration

        flow_iat = _iat(all_ts)
        fwd_iat = _iat([p.ts for p in fwd])
        bwd_iat = _iat([p.ts for p in bwd])

        f_iat_mean, f_iat_std, f_iat_max, f_iat_min = _stats(flow_iat)
        fw_iat_mean, fw_iat_std, fw_iat_max, fw_iat_min = _stats(fwd_iat)
        bw_iat_mean, bw_iat_std, bw_iat_max, bw_iat_min = _stats(bwd_iat)

        flag_counts = {"FIN": 0, "SYN": 0, "RST": 0, "PSH": 0, "ACK": 0, "URG": 0}
        for p in packets:
            for k in flag_counts:
                flag_counts[k] += p.flags.get(k, 0)

        down_up_ratio = (len(bwd) / len(fwd)) if len(fwd) > 0 else 0.0

        active_durs, idle_durs = _active_idle_periods(all_ts, self.idle_threshold)
        act_mean, act_std, act_max, act_min = _stats(active_durs)
        idl_mean, idl_std, idl_max, idl_min = _stats(idle_durs)

        return {
            "Flow Duration": duration,
            "Total Fwd Packets": len(fwd),
            "Total Backward Packets": len(bwd),
            "Total Length of Fwd Packets": sum(fwd_len),
            "Total Length of Bwd Packets": sum(bwd_len),
            "Fwd Packet Length Max": fwd_max,
            "Fwd Packet Length Min": fwd_min,
            "Fwd Packet Length Mean": fwd_mean,
            "Fwd Packet Length Std": fwd_std,
            "Bwd Packet Length Max": bwd_max,
            "Bwd Packet Length Min": bwd_min,
            "Bwd Packet Length Mean": bwd_mean,
            "Bwd Packet Length Std": bwd_std,
            "Flow Bytes/s": flow_bytes_per_s,
            "Flow Packets/s": flow_packets_per_s,
            "Flow IAT Mean": f_iat_mean,
            "Flow IAT Std": f_iat_std,
            "Flow IAT Max": f_iat_max,
            "Flow IAT Min": f_iat_min,
            "Fwd IAT Mean": fw_iat_mean,
            "Fwd IAT Std": fw_iat_std,
            "Fwd IAT Max": fw_iat_max,
            "Fwd IAT Min": fw_iat_min,
            "Bwd IAT Mean": bw_iat_mean,
            "Bwd IAT Std": bw_iat_std,
            "Bwd IAT Max": bw_iat_max,
            "Bwd IAT Min": bw_iat_min,
            "FIN Flag Count": flag_counts["FIN"],
            "SYN Flag Count": flag_counts["SYN"],
            "RST Flag Count": flag_counts["RST"],
            "PSH Flag Count": flag_counts["PSH"],
            "ACK Flag Count": flag_counts["ACK"],
            "URG Flag Count": flag_counts["URG"],
            "Down/Up Ratio": down_up_ratio,
            "Active Mean": act_mean,
            "Active Std": act_std,
            "Active Max": act_max,
            "Active Min": act_min,
            "Idle Mean": idl_mean,
            "Idle Std": idl_std,
            "Idle Max": idl_max,
            "Idle Min": idl_min,
        }


class WindowFeatureExtractor(threading.Thread):
    """
    1 thread rieng cho 1 kich thuoc sliding window (vd 1s, 3s, 5s).
    Doc doc lap tu FlowBuffer (khong dung chung Queue voi cac window khac)
    nen khong bi "dong bo cheo" hay tranh chap giua cac window.
    """

    def __init__(self, flow_buffer, out_queue: "queue.Queue", window_seconds: int,
                 extract_interval: float, idle_threshold: float, flow_timeout: float):
        super().__init__(name=f"Flow2-Window{window_seconds}s", daemon=True)
        self.flow_buffer = flow_buffer
        self.out_queue = out_queue
        self.window_seconds = window_seconds
        self.extract_interval = extract_interval
        self.flow_timeout = flow_timeout
        self.calculator = FeatureCalculator(idle_threshold)
        self._stop_event = threading.Event()

        self.processed_count = 0

    def run(self):
        logger.info("Flow2 worker (window=%ss) bat dau.", self.window_seconds)
        while not self._stop_event.is_set():
            start = time.time()
            active_keys = self.flow_buffer.active_flow_keys(self.flow_timeout)

            for flow_key in active_keys:
                packets = self.flow_buffer.snapshot_window(flow_key, self.window_seconds, now_ts=start)
                feats = self.calculator.compute(packets)
                if feats is None:
                    continue

                fv = FeatureVector(
                    flow_key=flow_key,
                    window_size=self.window_seconds,
                    ts=start,
                    values=feats,
                )
                try:
                    self.out_queue.put_nowait(fv)
                    self.processed_count += 1
                except queue.Full:
                    logger.warning("Queue2 dang day (window=%ss), bo qua 1 feature vector.",
                                   self.window_seconds)

            elapsed = time.time() - start
            sleep_time = max(0.0, self.extract_interval - elapsed)
            self._stop_event.wait(sleep_time)

        logger.info("Flow2 worker (window=%ss) da dung.", self.window_seconds)

    def stop(self):
        self._stop_event.set()