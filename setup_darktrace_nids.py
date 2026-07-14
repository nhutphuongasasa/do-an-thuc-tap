import os
import shutil

# 1. Define paths
src_model_dir = "/home/phuong/Documents/do an thuc tpa tot nghiep/TrafficGuard/models"
dest_root = "/home/phuong/Documents/do an thuc tpa tot nghiep/darktrace-nids"

subdirs = [
    "capture",
    "flow",
    "zeek",
    "suricata",
    "features",
    "ml",
    "ml/models",
    "correlation",
    "alert",
    "database",
    "queue",
    "logs",
    "tests"
]

print("Creating directories...")
os.makedirs(dest_root, exist_ok=True)
for d in subdirs:
    os.makedirs(os.path.join(dest_root, d), exist_ok=True)

# 2. Copy and rename model files
print("Copying model files...")
models_to_copy = [
    ("rf_final_model.pkl", "model.pkl"),
    ("scaler_final.pkl", "scaler.pkl"),
    ("label_encoder.pkl", "encoder.pkl")
]

for src_name, dest_name in models_to_copy:
    src_path = os.path.join(src_model_dir, src_name)
    dest_path = os.path.join(dest_root, "ml/models", dest_name)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dest_path)
        print(f"Copied {src_name} to ml/models/{dest_name}")
    else:
        print(f"Warning: {src_name} not found in {src_model_dir}")

# 3. Create __init__.py in all folders
print("Creating __init__.py files...")
for d in subdirs:
    # skip subdirectories if their parent already has __init__.py or if they are logs/tests
    if d in ["logs", "tests"]:
        continue
    init_path = os.path.join(dest_root, d, "__init__.py")
    with open(init_path, "w") as f:
        f.write(f'# Package {d}\n')

# 4. Write files helper
def write_file(rel_path, content):
    full_path = os.path.join(dest_root, rel_path)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"Wrote {rel_path}")

# ==========================================
# 5. Define file contents
# ==========================================

# requirements.txt
requirements_txt = """
scapy>=2.5.0
numpy
pandas
scikit-learn
joblib
requests
psycopg2-binary
python-dotenv
pytest
"""

# .env
env_content = """
# Database connection string
DATABASE_URL=postgresql://neondb_owner:npg_KBUNdnvg5W1s@ep-icy-dream-at0vl5nc-pooler.c-9.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require
DB_WRITE_ENABLED=true

# Capture configuration
CAPTURE_INTERFACE=
BPF_FILTER=ip

# Zeek & Suricata paths
SURICATA_EVE_JSON=logs/eve.json
ZEEK_LOG_PATH=logs/conn.log

# ML Configuration
BENIGN_LABEL=Benign
ALERT_THRESHOLD=30.0
"""

# config.py
config_py = """
import os
import multiprocessing
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # System
    CPU_COUNT = multiprocessing.cpu_count()
    LOG_LEVEL = "INFO"
    LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
    
    # Packet Capture
    CAPTURE_INTERFACE = os.environ.get("CAPTURE_INTERFACE", None)
    BPF_FILTER = os.environ.get("BPF_FILTER", "ip")
    QUEUE_PUT_TIMEOUT = 0.1
    QUEUE_GET_TIMEOUT = 0.5
    
    # Queues sizes
    QUEUE_MAXSIZE = 5000
    
    # Flow caching
    FLOW_IDLE_TIMEOUT = 25.0
    FLOW_HARD_TIMEOUT = 120.0
    FLOW_SWEEP_INTERVAL = 5.0
    
    # Models
    ML_MODEL_DIR = os.path.join(os.path.dirname(__file__), "ml", "models")
    ML_MODEL_PATH = os.path.join(ML_MODEL_DIR, "model.pkl")
    ML_SCALER_PATH = os.path.join(ML_MODEL_DIR, "scaler.pkl")
    ML_ENCODER_PATH = os.path.join(ML_MODEL_DIR, "encoder.pkl")
    
    BENIGN_LABEL = os.environ.get("BENIGN_LABEL", "Benign")
    
    # Suricata & Zeek
    SURICATA_EVE_JSON = os.environ.get("SURICATA_EVE_JSON", os.path.join(LOG_DIR, "eve.json"))
    ZEEK_LOG_PATH = os.environ.get("ZEEK_LOG_PATH", os.path.join(LOG_DIR, "conn.log"))
    
    # Database
    DATABASE_URL = os.environ.get("DATABASE_URL", None)
    DB_WRITE_ENABLED = os.environ.get("DB_WRITE_ENABLED", "true").lower() == "true"
    
    # Correlation & Scoring
    RULE_SCORE_WEIGHT = 0.6
    ML_SCORE_WEIGHT = 0.4
    RISK_THRESHOLD_LOW = 30.0
    RISK_THRESHOLD_MEDIUM = 60.0
    RISK_THRESHOLD_HIGH = 80.0
    ALERT_THRESHOLD = float(os.environ.get("ALERT_THRESHOLD", 30.0))
    
    # Alert Outputs
    ALERT_JSON_PATH = os.path.join(LOG_DIR, "alerts.json")
    ALERT_LOG_PATH = os.path.join(LOG_DIR, "alerts.log")

settings = Settings()
"""

# queue/event_queue.py
event_queue_py = """
import queue
import logging

logger = logging.getLogger("nids.queue")

class BoundedQueue:
    def __init__(self, name: str, maxsize: int = 5000):
        self.name = name
        self._q = queue.Queue(maxsize=maxsize)
        self.dropped_count = 0

    def put(self, item, timeout: float = 0.1) -> bool:
        try:
            self._q.put(item, timeout=timeout)
            return True
        except queue.Full:
            self.dropped_count += 1
            if self.dropped_count % 1000 == 1:
                logger.warning(f"Queue {self.name} is full! Dropped {self.dropped_count} items total.")
            return False

    def get(self, timeout: float = 0.5):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def qsize(self) -> int:
        return self._q.qsize()
"""

# capture/packet_parser.py
packet_parser_py = """
import time
import logging
from scapy.layers.inet import IP, TCP, UDP, ICMP

logger = logging.getLogger("nids.capture")

def parse_packet(pkt) -> dict | None:
    \"\"\"Parses a Scapy packet into a lightweight dictionary representation.\"\"\"
    if IP not in pkt:
        return None
    
    ip = pkt[IP]
    proto = "OTHER"
    src_port = 0
    dst_port = 0
    tcp_flags = {
        "FIN": 0, "SYN": 0, "RST": 0, "PSH": 0, "ACK": 0, "URG": 0
    }
    
    if TCP in pkt:
        proto = "TCP"
        src_port = int(pkt[TCP].sport)
        dst_port = int(pkt[TCP].dport)
        flags = int(pkt[TCP].flags)
        tcp_flags = {
            "FIN": int(bool(flags & 0x01)),
            "SYN": int(bool(flags & 0x02)),
            "RST": int(bool(flags & 0x04)),
            "PSH": int(bool(flags & 0x08)),
            "ACK": int(bool(flags & 0x10)),
            "URG": int(bool(flags & 0x20)),
        }
    elif UDP in pkt:
        proto = "UDP"
        src_port = int(pkt[UDP].sport)
        dst_port = int(pkt[UDP].dport)
    elif ICMP in pkt:
        proto = "ICMP"
        
    return {
        "timestamp": time.time(),
        "src_ip": ip.src,
        "dst_ip": ip.dst,
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": proto,
        "length": len(pkt),
        "tcp_flags": tcp_flags,
        "ttl": int(ip.ttl),
    }
"""

# capture/scapy_capture.py
scapy_capture_py = """
import logging
import threading
from scapy.all import sniff
from capture.packet_parser import parse_packet

logger = logging.getLogger("nids.capture")

class PacketCapture:
    def __init__(self, interface: str, bpf_filter: str, packet_queue, put_timeout: float = 0.1):
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.packet_queue = packet_queue
        self.put_timeout = put_timeout
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        logger.info(f"Starting PacketCapture on interface: {self.interface or 'default'}, filter: '{self.bpf_filter}'")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="CaptureThread", daemon=True)
        self._thread.start()

    def stop(self):
        logger.info("Stopping PacketCapture...")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("PacketCapture stopped.")

    def _run(self):
        try:
            sniff(
                iface=self.interface,
                filter=self.bpf_filter,
                prn=self._callback,
                store=False,
                stop_filter=lambda _: self._stop_event.is_set()
            )
        except PermissionError:
            logger.error("Permission denied opening interface. Elevated privileges (CAP_NET_RAW) required.")
        except Exception:
            logger.exception("Packet capture engine crashed")

    def _callback(self, packet):
        parsed = parse_packet(packet)
        if parsed:
            self.packet_queue.put(parsed, timeout=self.put_timeout)
"""

# flow/models.py
flow_models_py = """
from dataclasses import dataclass, field
import time

@dataclass(frozen=True)
class FlowKey:
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str

    def as_tuple(self):
        return (self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol)

@dataclass
class Flow:
    key: FlowKey
    start_time: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    initiator_ip: str = ""
    initiator_port: int = 0
    
    fwd_packets: list = field(default_factory=list)
    bwd_packets: list = field(default_factory=list)

    def add_packet(self, pkt: dict):
        self.last_seen = pkt.get("timestamp", time.time())
        if not self.initiator_ip:
            self.initiator_ip = pkt["src_ip"]
            self.initiator_port = pkt["src_port"]

        is_fwd = (pkt["src_ip"] == self.initiator_ip and pkt["src_port"] == self.initiator_port)
        if is_fwd:
            self.fwd_packets.append(pkt)
        else:
            self.bwd_packets.append(pkt)

    def duration(self) -> float:
        return max(0.0, self.last_seen - self.start_time)

    def to_dict(self) -> dict:
        return {
            "src_ip": self.key.src_ip,
            "dst_ip": self.key.dst_ip,
            "src_port": self.key.src_port,
            "dst_port": self.key.dst_port,
            "protocol": self.key.protocol,
            "start_time": self.start_time,
            "last_seen": self.last_seen,
            "duration": self.duration(),
            "fwd_packets": self.fwd_packets,
            "bwd_packets": self.bwd_packets,
        }
"""

# flow/flow_cache.py
flow_cache_py = """
import threading
import time
from flow.models import Flow, FlowKey

class FlowCache:
    def __init__(self, idle_timeout: float, hard_timeout: float):
        self.idle_timeout = idle_timeout
        self.hard_timeout = hard_timeout
        self._flows: dict[tuple, Flow] = {}
        self._lock = threading.Lock()

    def get_or_create(self, key: FlowKey) -> Flow:
        with self._lock:
            # Sort IP and ports to make it bidirectional
            ip1, ip2 = key.src_ip, key.dst_ip
            p1, p2 = key.src_port, key.dst_port
            if (ip1, p1) > (ip2, p2):
                ip1, ip2 = ip2, ip1
                p1, p2 = p2, p1
            
            t = (ip1, ip2, p1, p2, key.protocol)
            flow = self._flows.get(t)
            if flow is None:
                flow = Flow(key=key)
                self._flows[t] = flow
            return flow

    def sweep_expired(self) -> list[Flow]:
        now = time.time()
        expired = []
        with self._lock:
            for t, flow in list(self._flows.items()):
                idle_for = now - flow.last_seen
                age = now - flow.start_time
                if idle_for >= self.idle_timeout or age >= self.hard_timeout:
                    expired.append(flow)
                    del self._flows[t]
        return expired

    def active_count(self) -> int:
        with self._lock:
            return len(self._flows)
"""

# flow/flow_manager.py
flow_manager_py = """
import logging
import threading
import time
from flow.flow_cache import FlowCache
from flow.models import FlowKey

logger = logging.getLogger("nids.flow")

class FlowManager:
    def __init__(self, idle_timeout: float, hard_timeout: float):
        self.cache = FlowCache(idle_timeout=idle_timeout, hard_timeout=hard_timeout)

    def process_packet(self, pkt: dict):
        key = FlowKey(
            src_ip=pkt["src_ip"],
            dst_ip=pkt["dst_ip"],
            src_port=pkt["src_port"],
            dst_port=pkt["dst_port"],
            protocol=pkt["protocol"],
        )
        flow = self.cache.get_or_create(key)
        flow.add_packet(pkt)
        return flow

    def flush_timeout_flows(self) -> list[dict]:
        expired = self.cache.sweep_expired()
        return [f.to_dict() for f in expired]

    def active_flow_count(self) -> int:
        return self.cache.active_count()

class FlowWorker:
    def __init__(self, worker_id: int, packet_queue, flow_queue,
                 idle_timeout: float, hard_timeout: float, sweep_interval: float,
                 get_timeout: float = 0.5, put_timeout: float = 0.1):
        self.worker_id = worker_id
        self.packet_queue = packet_queue
        self.flow_queue = flow_queue
        self.sweep_interval = sweep_interval
        self.get_timeout = get_timeout
        self.put_timeout = put_timeout
        self.manager = FlowManager(idle_timeout=idle_timeout, hard_timeout=hard_timeout)
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=f"FlowWorker-{self.worker_id}", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        logger.info(f"FlowWorker-{self.worker_id} started")
        last_sweep = time.time()
        while not self._stop_event.is_set():
            pkt = self.packet_queue.get(timeout=self.get_timeout)
            if pkt:
                self.manager.process_packet(pkt)
            
            if time.time() - last_sweep >= self.sweep_interval:
                self.flush_timeout_flows()
                last_sweep = time.time()
                
        # Final flush on shutdown
        self.flush_timeout_flows()
        logger.info(f"FlowWorker-{self.worker_id} stopped")

    def flush_timeout_flows(self):
        expired = self.manager.flush_timeout_flows()
        for f in expired:
            self.flow_queue.put(f, timeout=self.put_timeout)
"""

# features/feature_schema.py
feature_schema_py = """
FEATURE_FIELDS = [
    'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Fwd Packet Length Max', 'Bwd Packet Length Max',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Mean', 'Fwd IAT Std',
    'Bwd IAT Mean', 'Bwd IAT Std', 'Down/Up Ratio', 'Active Mean', 'Active Std', 'Idle Mean', 'Idle Std',
    'PSH Flag Count', 'SYN Flag Count', 'RST Flag Count', 'Flow Duration', 'Total Fwd Packets',
    'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Min', 'Bwd Packet Length Min', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Max',
    'Bwd IAT Min', 'FIN Flag Count', 'ACK Flag Count', 'URG Flag Count', 'Active Max', 'Active Min',
    'Idle Max', 'Idle Min'
]

def as_vector(feature_dict: dict) -> list:
    return [float(feature_dict.get(f, 0.0)) for f in FEATURE_FIELDS]
"""

# features/feature_extractor.py
feature_extractor_py = """
import logging
import threading
import time
import numpy as np
from features.feature_schema import FEATURE_FIELDS

logger = logging.getLogger("nids.features")

def _stats(values):
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype="float64")
    return float(arr.mean()), float(arr.std()), float(arr.max()), float(arr.min())

def _iat(timestamps):
    if len(timestamps) < 2:
        return []
    ts = sorted(timestamps)
    return [ts[i] - ts[i - 1] for i in range(1, len(ts))]

def _active_idle_periods(timestamps, idle_threshold=1.0):
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

class FeatureExtractor:
    def __init__(self, idle_threshold: float = 1.0):
        self.idle_threshold = idle_threshold

    def extract(self, flow: dict) -> dict:
        fwd = flow.get("fwd_packets", [])
        bwd = flow.get("bwd_packets", [])
        
        all_packets = sorted(fwd + bwd, key=lambda p: p["timestamp"])
        all_ts = [p["timestamp"] for p in all_packets]
        duration = flow.get("duration", 0.0)
        duration = max(duration, 1e-6)

        fwd_len = [p["length"] for p in fwd]
        bwd_len = [p["length"] for p in bwd]

        fwd_mean, fwd_std, fwd_max, fwd_min = _stats(fwd_len)
        bwd_mean, bwd_std, bwd_max, bwd_min = _stats(bwd_len)

        total_bytes = sum(fwd_len) + sum(bwd_len)
        flow_bytes_per_s = total_bytes / duration
        flow_packets_per_s = len(all_packets) / duration

        flow_iat = _iat(all_ts)
        fwd_iat = _iat([p["timestamp"] for p in fwd])
        bwd_iat = _iat([p["timestamp"] for p in bwd])

        f_iat_mean, f_iat_std, f_iat_max, f_iat_min = _stats(flow_iat)
        fw_iat_mean, fw_iat_std, fw_iat_max, fw_iat_min = _stats(fwd_iat)
        bw_iat_mean, bw_iat_std, bw_iat_max, bw_iat_min = _stats(bwd_iat)

        flag_counts = {"FIN": 0, "SYN": 0, "RST": 0, "PSH": 0, "ACK": 0, "URG": 0}
        for p in all_packets:
            flags = p.get("tcp_flags") or {}
            for k in flag_counts:
                flag_counts[k] += flags.get(k, 0)

        down_up_ratio = (len(bwd) / len(fwd)) if len(fwd) > 0 else 0.0

        active_durs, idle_durs = _active_idle_periods(all_ts, self.idle_threshold)
        act_mean, act_std, act_max, act_min = _stats(active_durs)
        idl_mean, idl_std, idl_max, idl_min = _stats(idle_durs)

        feature = {
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

        feature["_meta"] = {
            "ts": flow.get("last_seen"),
            "src_ip": flow.get("src_ip"),
            "dst_ip": flow.get("dst_ip"),
            "src_port": flow.get("src_port"),
            "dst_port": flow.get("dst_port"),
            "protocol": flow.get("protocol"),
        }
        return feature

class FeatureWorker:
    def __init__(self, worker_id: int, flow_queue, feature_queue,
                 get_timeout: float = 0.5, put_timeout: float = 0.1):
        self.worker_id = worker_id
        self.flow_queue = flow_queue
        self.feature_queue = feature_queue
        self.get_timeout = get_timeout
        self.put_timeout = put_timeout
        self.extractor = FeatureExtractor()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name=f"FeatureWorker-{self.worker_id}", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        logger.info(f"FeatureWorker-{self.worker_id} started")
        while not self._stop_event.is_set():
            flow = self.flow_queue.get(timeout=self.get_timeout)
            if flow:
                try:
                    feature = self.extractor.extract(flow)
                    self.feature_queue.put(feature, timeout=self.put_timeout)
                except Exception:
                    logger.exception(f"FeatureWorker-{self.worker_id} failed to extract features")
        logger.info(f"FeatureWorker-{self.worker_id} stopped")
"""

# ml/model_loader.py
ml_model_loader_py = """
import os
import joblib
import logging

logger = logging.getLogger("nids.ml")

class ModelLoader:
    @staticmethod
    def load(model_path: str, scaler_path: str, encoder_path: str):
        model, scaler, encoder = None, None, None
        
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                logger.info("Loaded model from %s", model_path)
            except Exception:
                logger.exception("Failed to load model from %s", model_path)
                
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
                logger.info("Loaded scaler from %s", scaler_path)
            except Exception:
                logger.exception("Failed to load scaler from %s", scaler_path)
                
        if os.path.exists(encoder_path):
            try:
                encoder = joblib.load(encoder_path)
                logger.info("Loaded encoder from %s", encoder_path)
            except Exception:
                logger.exception("Failed to load encoder from %s", encoder_path)
                
        return model, scaler, encoder
"""

# ml/inference.py
ml_inference_py = """
import logging
import multiprocessing
import time
import pandas as pd
from features.feature_schema import as_vector, FEATURE_FIELDS

logger = logging.getLogger("nids.ml")

class MLInference:
    def __init__(self, model, scaler, encoder):
        self.model = model
        self.scaler = scaler
        self.encoder = encoder

    def predict(self, feature: dict) -> dict:
        if self.model is None or self.scaler is None:
            return {"attack_label": "Benign", "confidence": 1.0}

        vector = as_vector(feature)
        
        try:
            # Recreate DataFrame with original feature names to satisfy Scaler/Model expectations
            df = pd.DataFrame([vector], columns=FEATURE_FIELDS)
            scaled = self.scaler.transform(df)
            
            # Predict
            pred_idx = self.model.predict(scaled)[0]
            
            # Get probability
            confidence = 1.0
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(scaled)[0]
                confidence = float(max(proba))
            
            # Decode label
            if self.encoder:
                attack_label = self.encoder.inverse_transform([pred_idx])[0]
            else:
                attack_label = str(pred_idx)
                
            return {"attack_label": attack_label, "confidence": round(confidence, 4)}
            
        except Exception:
            logger.exception("ML inference failed, defaulting to Benign")
            return {"attack_label": "Benign", "confidence": 1.0}

def _ml_worker_main(worker_id: int, model_path: str, scaler_path: str, encoder_path: str,
                     feature_queue, ml_result_queue, stop_event, get_timeout: float, put_timeout: float, log_level: str):
    import logging
    logging.basicConfig(level=log_level, format=f"%(asctime)s [ML-{worker_id}] %(levelname)s %(message)s")
    log = logging.getLogger(f"nids.worker.ml.{worker_id}")

    from ml.model_loader import ModelLoader
    model, scaler, encoder = ModelLoader.load(model_path, scaler_path, encoder_path)
    inference = MLInference(model, scaler, encoder)
    log.info(f"MLWorker-{worker_id} child process ready.")

    while not stop_event.is_set():
        feature = feature_queue.get(timeout=get_timeout)
        if feature is None:
            continue
        try:
            result = inference.predict(feature)
            result["_meta"] = feature.get("_meta", {})
            result["worker_id"] = worker_id
            ml_result_queue.put(result, timeout=put_timeout)
        except Exception:
            log.exception("ML worker failed on inference")
    log.info(f"MLWorker-{worker_id} child process stopped.")

class MLWorker:
    def __init__(self, worker_id: int, model_path: str, scaler_path: str, encoder_path: str,
                 feature_queue, ml_result_queue, get_timeout: float = 0.5, put_timeout: float = 0.1, log_level: str = "INFO"):
        self.worker_id = worker_id
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoder_path = encoder_path
        self.feature_queue = feature_queue
        self.ml_result_queue = ml_result_queue
        self.get_timeout = get_timeout
        self.put_timeout = put_timeout
        self.log_level = log_level

        self._stop_event = multiprocessing.Event()
        self._process = multiprocessing.Process(
            target=_ml_worker_main,
            args=(
                worker_id,
                model_path,
                scaler_path,
                encoder_path,
                feature_queue,
                ml_result_queue,
                self._stop_event,
                get_timeout,
                put_timeout,
                log_level
            ),
            name=f"MLWorker-{worker_id}",
            daemon=True
        )

    def start(self):
        self._process.start()
        logger.info(f"MLWorker-{self.worker_id} process started (PID: {self._process.pid})")

    def stop(self):
        self._stop_event.set()
        self._process.join(timeout=5)
        if self._process.is_alive():
            logger.warning(f"MLWorker-{self.worker_id} did not exit cleanly, terminating...")
            self._process.terminate()
        logger.info(f"MLWorker-{self.worker_id} stopped.")
"""

# zeek/zeek_parser.py
zeek_parser_py = """
import json
import logging
import os
import time

logger = logging.getLogger("nids.zeek")

class ZeekParser:
    def __init__(self, log_path: str, poll_interval: float = 1.0):
        self.log_path = log_path
        self.poll_interval = poll_interval
        self._stop = False
        self._fh = None

    def stop(self):
        self._stop = True

    def _open(self):
        if not os.path.exists(self.log_path):
            return None
        fh = open(self.log_path, "r")
        fh.seek(0, os.SEEK_END)
        return fh

    def tail(self):
        while not self._stop:
            if self._fh is None:
                self._fh = self._open()
                if self._fh is None:
                    time.sleep(self.poll_interval)
                    continue

            line = self._fh.readline()
            if not line:
                time.sleep(self.poll_interval)
                continue

            record = self._parse_line(line)
            if record:
                yield record

    @staticmethod
    def _parse_line(line: str) -> dict | None:
        line = line.strip()
        if not line:
            return None
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            return None

        return {
            "ts": raw.get("ts"),
            "uid": raw.get("uid"),
            "src_ip": raw.get("id.orig_h"),
            "src_port": raw.get("id.orig_p"),
            "dst_ip": raw.get("id.resp_h"),
            "dst_port": raw.get("id.resp_p"),
            "protocol": (raw.get("proto") or "").upper(),
            "service": raw.get("service"),
            "duration": raw.get("duration"),
            "orig_bytes": raw.get("orig_bytes"),
            "resp_bytes": raw.get("resp_bytes"),
            "conn_state": raw.get("conn_state"),
            "orig_pkts": raw.get("orig_pkts"),
            "resp_pkts": raw.get("resp_pkts"),
        }
"""

# suricata/rule_event.py
suricata_rule_event_py = """
from dataclasses import dataclass

SEVERITY_TO_SCORE = {
    1: 90,
    2: 60,
    3: 30,
}

@dataclass
class RuleEvent:
    signature: str
    severity: int
    category: str
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: str
    timestamp: str

    @property
    def rule_score(self) -> int:
        return SEVERITY_TO_SCORE.get(self.severity, 50)

    def to_dict(self) -> dict:
        return {
            "signature": self.signature,
            "severity": self.severity,
            "category": self.category,
            "source_ip": self.source_ip,
            "destination_ip": self.destination_ip,
            "source_port": self.source_port,
            "destination_port": self.destination_port,
            "protocol": self.protocol,
            "timestamp": self.timestamp,
            "rule_score": self.rule_score,
        }
"""

# suricata/eve_reader.py
suricata_eve_reader_py = """
import json
import logging
import os
import time
from suricata.rule_event import RuleEvent

logger = logging.getLogger("nids.suricata")

class SuricataEveReader:
    def __init__(self, eve_path: str, poll_interval: float = 0.5):
        self.eve_path = eve_path
        self.poll_interval = poll_interval
        self._stop = False
        self._fh = None

    def stop(self):
        self._stop = True

    def _open(self):
        if not os.path.exists(self.eve_path):
            return None
        fh = open(self.eve_path, "r")
        fh.seek(0, os.SEEK_END)
        return fh

    def tail(self):
        while not self._stop:
            if self._fh is None:
                self._fh = self._open()
                if self._fh is None:
                    time.sleep(self.poll_interval)
                    continue

            line = self._fh.readline()
            if not line:
                time.sleep(self.poll_interval)
                continue

            event = self._parse_line(line)
            if event:
                yield event

    @staticmethod
    def _parse_line(line: str) -> RuleEvent | None:
        line = line.strip()
        if not line:
            return None
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            return None

        if raw.get("event_type") != "alert":
            return None

        alert = raw.get("alert", {})
        try:
            return RuleEvent(
                signature=alert.get("signature", "unknown"),
                severity=int(alert.get("severity", 3)),
                category=alert.get("category", "unknown"),
                source_ip=raw.get("src_ip", ""),
                destination_ip=raw.get("dest_ip", ""),
                source_port=int(raw.get("src_port", 0) or 0),
                destination_port=int(raw.get("dest_port", 0) or 0),
                protocol=raw.get("proto", ""),
                timestamp=raw.get("timestamp", ""),
            )
        except Exception:
            logger.debug("Failed to parse Suricata alert line", exc_info=True)
            return None
"""

# correlation/scoring.py
correlation_scoring_py = """
def map_score_to_severity(score: float, low=30.0, medium=60.0, high=80.0) -> str:
    if score < low:
        return "Normal"
    elif score < medium:
        return "Low"
    elif score < high:
        return "Medium"
    return "Critical"
"""

# correlation/risk_engine.py
correlation_risk_engine_py = """
import logging
from correlation.scoring import map_score_to_severity

logger = logging.getLogger("nids.risk")

class RiskEngine:
    def __init__(self, rule_weight: float = 0.6, ml_weight: float = 0.4,
                 low: float = 30.0, medium: float = 60.0, high: float = 80.0):
        self.rule_weight = rule_weight
        self.ml_weight = ml_weight
        self.low = low
        self.medium = medium
        self.high = high

    def correlate(self, rule_event: dict | None, ml_result: dict | None, context: dict) -> dict:
        rule_score = rule_event.get("rule_score", 0.0) if rule_event else 0.0
        
        ml_score = 0.0
        if ml_result:
            if ml_result.get("attack_label") != "Benign":
                ml_score = float(ml_result.get("confidence", 0.0)) * 100.0

        if rule_event and ml_result and ml_result.get("attack_label") != "Benign":
            risk_score = (rule_score * self.rule_weight) + (ml_score * self.ml_weight)
        elif rule_event:
            risk_score = rule_score
        elif ml_result and ml_result.get("attack_label") != "Benign":
            risk_score = ml_score
        else:
            risk_score = 0.0

        severity = map_score_to_severity(risk_score, self.low, self.medium, self.high)

        return {
            "timestamp": ml_result.get("_meta", {}).get("ts") if ml_result else (rule_event.get("timestamp") if rule_event else None),
            "risk_score": risk_score,
            "severity": severity,
            "context": context,
            "ml_evidence": ml_result,
            "rule_evidence": rule_event
        }
"""

# alert/json_writer.py
alert_json_writer_py = """
import json
import os

class JSONWriter:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def write(self, event: dict):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\\n")
"""

# alert/alert_manager.py
alert_alert_manager_py = """
import os
import logging
from alert.json_writer import JSONWriter

logger = logging.getLogger("nids.alert")

class AlertManager:
    def __init__(self, json_path: str, log_path: str, threshold: float = 30.0, db_writer=None):
        self.json_writer = JSONWriter(json_path)
        self.log_path = log_path
        self.threshold = threshold
        self.db_writer = db_writer
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def handle(self, risk_event: dict):
        score = risk_event.get("risk_score", 0.0)
        if score < self.threshold:
            return

        # Write to JSON file
        self.json_writer.write(risk_event)

        # Write to log file
        ctx = risk_event.get("context", {})
        ml = risk_event.get("ml_evidence") or {}
        rule = risk_event.get("rule_evidence") or {}
        
        msg = (
            f"ALERT | Severity: {risk_event['severity']} | Risk Score: {score:.1f} | "
            f"Flow: {ctx.get('src_ip')}:{ctx.get('src_port')} -> {ctx.get('dst_ip')}:{ctx.get('dst_port')} ({ctx.get('protocol')}) | "
            f"ML: {ml.get('attack_label')} ({ml.get('confidence')}) | "
            f"Rule: {rule.get('signature', 'None')}"
        )
        
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\\n")
            
        logger.warning(msg)

        # Write to database if configured
        if self.db_writer:
            try:
                self.db_writer.insert_alert(risk_event)
            except Exception:
                logger.exception("Failed to write alert to PostgreSQL database")
"""

# database/postgres.py
database_postgres_py = """
import logging
import psycopg2
import psycopg2.extras

logger = logging.getLogger("nids.db")

INSERT_SQL = \"\"\"
INSERT INTO alerts (
    ts, risk_score, severity, src_ip, dst_ip, src_port, dst_port,
    protocol, attack_label, ml_confidence, rule_signature, rule_severity,
    raw_event
) VALUES (
    to_timestamp(%(ts)s), %(risk_score)s, %(severity)s, %(src_ip)s, %(dst_ip)s,
    %(src_port)s, %(dst_port)s, %(protocol)s, %(attack_label)s,
    %(ml_confidence)s, %(rule_signature)s, %(rule_severity)s, %(raw_event)s
)
\"\"\"

class PostgresWriter:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._conn = None

    def connect(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.database_url)
            self._conn.autocommit = True
            logger.info("Connected to PostgreSQL database successfully.")
        return self._conn

    def insert_alert(self, risk_event: dict):
        conn = self.connect()
        ctx = risk_event.get("context", {})
        ml = risk_event.get("ml_evidence") or {}
        rule = risk_event.get("rule_evidence") or {}

        # Fallback timestamp
        import time
        ts = risk_event.get("timestamp") or time.time()
        if isinstance(ts, str):
            try:
                from dateutil.parser import parse
                ts = parse(ts).timestamp()
            except Exception:
                ts = time.time()

        params = {
            "ts": ts,
            "risk_score": risk_event.get("risk_score"),
            "severity": risk_event.get("severity"),
            "src_ip": ctx.get("src_ip"),
            "dst_ip": ctx.get("dst_ip"),
            "src_port": ctx.get("src_port"),
            "dst_port": ctx.get("dst_port"),
            "protocol": ctx.get("protocol"),
            "attack_label": ml.get("attack_label"),
            "ml_confidence": ml.get("confidence"),
            "rule_signature": rule.get("signature"),
            "rule_severity": rule.get("severity"),
            "raw_event": psycopg2.extras.Json(risk_event),
        }

        with conn.cursor() as cur:
            cur.execute(INSERT_SQL, params)

    def close(self):
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
"""

# database/schema.sql
database_schema_sql = """
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMP WITH TIME ZONE NOT NULL,
    risk_score FLOAT NOT NULL,
    severity VARCHAR(20) NOT NULL,
    src_ip VARCHAR(45) NOT NULL,
    dst_ip VARCHAR(45) NOT NULL,
    src_port INT NOT NULL,
    dst_port INT NOT NULL,
    protocol VARCHAR(10) NOT NULL,
    attack_label VARCHAR(100),
    ml_confidence FLOAT,
    rule_signature VARCHAR(255),
    rule_severity INT,
    raw_event JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

# worker_manager.py
worker_manager_py = """
import logging
import threading
import time
from config import settings
from queue.event_queue import BoundedQueue
from capture.scapy_capture import PacketCapture
from flow.flow_manager import FlowWorker
from features.feature_extractor import FeatureWorker
from ml.inference import MLWorker
from suricata.eve_reader import SuricataEveReader
from correlation.risk_engine import RiskEngine
from alert.alert_manager import AlertManager

logger = logging.getLogger("nids.worker_manager")

class WorkerManager:
    def __init__(self):
        # 1. Initialize queues
        self.packet_queue = BoundedQueue("PacketQueue", settings.QUEUE_MAXSIZE)
        self.flow_queue = BoundedQueue("FlowQueue", settings.QUEUE_MAXSIZE)
        self.feature_queue = BoundedQueue("FeatureQueue", settings.QUEUE_MAXSIZE)
        self.ml_result_queue = BoundedQueue("MLResultQueue", settings.QUEUE_MAXSIZE)
        
        # 2. Packet Capture
        self.capture_worker = PacketCapture(
            interface=settings.CAPTURE_INTERFACE,
            bpf_filter=settings.BPF_FILTER,
            packet_queue=self.packet_queue,
            put_timeout=settings.QUEUE_PUT_TIMEOUT
        )
        
        # 3. Flow Worker Pool
        self.flow_workers = [
            FlowWorker(
                worker_id=i,
                packet_queue=self.packet_queue,
                flow_queue=self.flow_queue,
                idle_timeout=settings.FLOW_IDLE_TIMEOUT,
                hard_timeout=settings.FLOW_HARD_TIMEOUT,
                sweep_interval=settings.FLOW_SWEEP_INTERVAL,
                get_timeout=settings.QUEUE_GET_TIMEOUT,
                put_timeout=settings.QUEUE_PUT_TIMEOUT
            )
            for i in range(2)  # Fixed pool size of 2 for flow tracking
        ]
        
        # 4. Feature Worker Pool
        self.feature_workers = [
            FeatureWorker(
                worker_id=i,
                flow_queue=self.flow_queue,
                feature_queue=self.feature_queue,
                get_timeout=settings.QUEUE_GET_TIMEOUT,
                put_timeout=settings.QUEUE_PUT_TIMEOUT
            )
            for i in range(2)
        ]
        
        # 5. ML Worker Pool
        self.ml_workers = [
            MLWorker(
                worker_id=i,
                model_path=settings.ML_MODEL_PATH,
                scaler_path=settings.ML_SCALER_PATH,
                encoder_path=settings.ML_ENCODER_PATH,
                feature_queue=self.feature_queue,
                ml_result_queue=self.ml_result_queue,
                get_timeout=settings.QUEUE_GET_TIMEOUT,
                put_timeout=settings.QUEUE_PUT_TIMEOUT,
                log_level=settings.LOG_LEVEL
            )
            for i in range(2)
        ]
        
        # 6. Suricata
        self.suricata_reader = SuricataEveReader(settings.SURICATA_EVE_JSON)
        self._recent_rule_events = {}
        self._rule_lock = threading.Lock()
        
        # 7. Risk Engine & Alert Manager
        self.risk_engine = RiskEngine(
            rule_weight=settings.RULE_SCORE_WEIGHT,
            ml_weight=settings.ML_SCORE_WEIGHT,
            low=settings.RISK_THRESHOLD_LOW,
            medium=settings.RISK_THRESHOLD_MEDIUM,
            high=settings.RISK_THRESHOLD_HIGH
        )
        
        db_writer = None
        if settings.DB_WRITE_ENABLED and settings.DATABASE_URL:
            try:
                from database.postgres import PostgresWriter
                db_writer = PostgresWriter(settings.DATABASE_URL)
                db_writer.connect()
            except Exception:
                logger.exception("Could not connect to PostgreSQL, running without DB writer.")
                
        self.alert_manager = AlertManager(
            json_path=settings.ALERT_JSON_PATH,
            log_path=settings.ALERT_LOG_PATH,
            threshold=settings.ALERT_THRESHOLD,
            db_writer=db_writer
        )
        
        self._stop_event = threading.Event()
        self._suricata_thread = None
        self._correlation_thread = None

    def start(self):
        logger.info("Starting NIDS Pipeline components...")
        self._stop_event.clear()
        
        # Start ML workers (Process pool)
        for w in self.ml_workers:
            w.start()
            
        # Start Feature workers (Threads)
        for w in self.feature_workers:
            w.start()
            
        # Start Flow workers (Threads)
        for w in self.flow_workers:
            w.start()
            
        # Start background threads for Suricata and Correlation
        self._suricata_thread = threading.Thread(target=self._suricata_loop, name="SuricataThread", daemon=True)
        self._suricata_thread.start()
        
        self._correlation_thread = threading.Thread(target=self._correlation_loop, name="CorrelationThread", daemon=True)
        self._correlation_thread.start()
        
        # Start packet capture
        self.capture_worker.start()
        logger.info("NIDS Pipeline components started successfully.")

    def stop(self):
        logger.info("Stopping NIDS Pipeline components...")
        self._stop_event.set()
        
        # Stop capture first
        self.capture_worker.stop()
        
        # Stop flow tracking
        for w in self.flow_workers:
            w.stop()
            
        # Stop feature extraction
        for w in self.feature_workers:
            w.stop()
            
        # Stop ML Inference
        for w in self.ml_workers:
            w.stop()
            
        # Stop Suricata Reader
        self.suricata_reader.stop()
        
        if self._suricata_thread:
            self._suricata_thread.join(timeout=5)
        if self._correlation_thread:
            self._correlation_thread.join(timeout=5)
            
        # Close database connection
        if self.alert_manager.db_writer:
            self.alert_manager.db_writer.close()
            
        logger.info("NIDS Pipeline stopped.")

    def _suricata_loop(self):
        try:
            for event in self.suricata_reader.tail():
                if self._stop_event.is_set():
                    break
                key = (
                    event.source_ip,
                    event.destination_ip,
                    event.source_port,
                    event.destination_port,
                    event.protocol.upper()
                )
                with self._rule_lock:
                    self._recent_rule_events[key] = event.to_dict()
        except Exception:
            logger.exception("Suricata reader thread crashed")

    def _lookup_rule_event(self, context: dict):
        key = (
            context.get("src_ip"),
            context.get("dst_ip"),
            context.get("src_port"),
            context.get("dst_port"),
            (context.get("protocol") or "").upper()
        )
        with self._rule_lock:
            # Pop so we don't keep growing memory indefinitely
            return self._recent_rule_events.pop(key, None)

    def _correlation_loop(self):
        while not self._stop_event.is_set():
            ml_result = self.ml_result_queue.get(timeout=settings.QUEUE_GET_TIMEOUT)
            if ml_result is None:
                continue
            
            try:
                context = ml_result.get("_meta", {})
                rule_event = self._lookup_rule_event(context)
                risk_event = self.risk_engine.correlate(rule_event, ml_result, context)
                self.alert_manager.handle(risk_event)
            except Exception:
                logger.exception("Failed to correlate ML result and Rule event")
"""

# main.py
main_py = """
import logging
import os
import sys
import time
import signal
from config import settings
from worker_manager import WorkerManager

def setup_logging():
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    log_file = os.path.join(settings.LOG_DIR, "nids.log")
    
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger("nids.main")
    
    logger.info("=" * 60)
    logger.info("Starting Modular Network Intrusion Detection System (NIDS/NDR)")
    logger.info("=" * 60)
    
    manager = WorkerManager()
    shutdown_requested = {"flag": False}
    
    def _handle_signal(signum, _frame):
        if shutdown_requested["flag"]:
            logger.warning("Force exiting...")
            sys.exit(1)
        logger.info(f"Received signal {signum}, starting graceful shutdown...")
        shutdown_requested["flag"] = True
        
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    
    manager.start()
    
    try:
        while not shutdown_requested["flag"]:
            time.sleep(1)
    finally:
        manager.stop()
        logger.info("NIDS Shutdown completed.")

if __name__ == "__main__":
    main()
"""

# tests/test_flow_and_features.py
tests_flow_and_features_py = """
import time
import sys
import os

# Insert parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.flow_manager import FlowManager
from features.feature_extractor import FeatureExtractor
from correlation.risk_engine import RiskEngine

def make_packet(src_ip="10.0.0.1", dst_ip="10.0.0.2", sport=1111, dport=80,
                proto="TCP", length=100, flags=None):
    if flags is None:
        flags = {"FIN": 0, "SYN": 1, "RST": 0, "PSH": 0, "ACK": 0, "URG": 0}
    return {
        "timestamp": time.time(),
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": sport,
        "dst_port": dport,
        "protocol": proto,
        "length": length,
        "tcp_flags": flags,
        "ttl": 64,
    }

def test_flow_manager_aggregates_5_tuple():
    fm = FlowManager(idle_timeout=1, hard_timeout=10)
    for _ in range(5):
        fm.process_packet(make_packet())
    assert fm.active_flow_count() == 1

    # Different port makes it a new flow
    fm.process_packet(make_packet(dport=443))
    assert fm.active_flow_count() == 2

def test_flow_manager_flushes_idle_flows():
    fm = FlowManager(idle_timeout=0.01, hard_timeout=10)
    fm.process_packet(make_packet())
    time.sleep(0.05)
    completed = fm.flush_timeout_flows()
    assert len(completed) == 1
    assert fm.active_flow_count() == 0

def test_feature_extractor_produces_42_fields():
    fm = FlowManager(idle_timeout=0.01, hard_timeout=10)
    fm.process_packet(make_packet(length=100))
    fm.process_packet(make_packet(src_ip="10.0.0.2", dst_ip="10.0.0.1", sport=80, dport=1111, length=200)) # backward
    time.sleep(0.05)
    completed = fm.flush_timeout_flows()
    assert len(completed) == 1

    extractor = FeatureExtractor()
    feature = extractor.extract(completed[0])
    
    # Check some critical fields
    assert "Flow Duration" in feature
    assert feature["Total Fwd Packets"] == 1
    assert feature["Total Backward Packets"] == 1
    assert feature["Total Length of Fwd Packets"] == 100
    assert feature["Total Length of Bwd Packets"] == 200

def test_risk_engine_combines_rule_and_ml_scores():
    engine = RiskEngine(rule_weight=0.6, ml_weight=0.4, low=30, medium=60, high=80)

    rule_event = {"rule_score": 90}
    ml_result = {"attack_label": "PortScan", "confidence": 0.8}

    result = engine.correlate(rule_event, ml_result, context={"src_ip": "1.2.3.4"})
    expected = (90 * 0.6) + (80.0 * 0.4) # confidence 0.8 -> ml_score = 80
    assert abs(result["risk_score"] - expected) < 0.01
    assert result["severity"] in ("Low", "Medium", "High", "Critical")

if __name__ == "__main__":
    test_flow_manager_aggregates_5_tuple()
    test_flow_manager_flushes_idle_flows()
    test_feature_extractor_produces_42_fields()
    test_risk_engine_combines_rule_and_ml_scores()
    print("All tests passed!")
"""

# ==========================================
# 6. Write all files
# ==========================================
write_file("requirements.txt", requirements_txt)
write_file(".env", env_content)
write_file("config.py", config_py)
write_file("queue/event_queue.py", event_queue_py)
write_file("capture/packet_parser.py", packet_parser_py)
write_file("capture/scapy_capture.py", scapy_capture_py)
write_file("flow/models.py", flow_models_py)
write_file("flow/flow_cache.py", flow_cache_py)
write_file("flow/flow_manager.py", flow_manager_py)
write_file("features/feature_schema.py", feature_schema_py)
write_file("features/feature_extractor.py", feature_extractor_py)
write_file("ml/model_loader.py", ml_model_loader_py)
write_file("ml/inference.py", ml_inference_py)
write_file("zeek/zeek_parser.py", zeek_parser_py)
write_file("suricata/rule_event.py", suricata_rule_event_py)
write_file("suricata/eve_reader.py", suricata_eve_reader_py)
write_file("correlation/scoring.py", correlation_scoring_py)
write_file("correlation/risk_engine.py", correlation_risk_engine_py)
write_file("alert/json_writer.py", alert_json_writer_py)
write_file("alert/alert_manager.py", alert_alert_manager_py)
write_file("database/postgres.py", database_postgres_py)
write_file("database/schema.sql", database_schema_sql)
write_file("worker_manager.py", worker_manager_py)
write_file("main.py", main_py)
write_file("tests/test_flow_and_features.py", tests_flow_and_features_py)

print("Darktrace modular NIDS folder setup complete!")
