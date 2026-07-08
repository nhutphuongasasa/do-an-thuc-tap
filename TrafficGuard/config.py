import os

# ==================== FLOW 1 - Packet Capture ====================
# None = de scapy tu chon interface mac dinh cua OS.
# Vi du Linux: "eth0" | Windows: r"\Device\NPF_{GUID}" hoac ten hien trong `scapy -c` / Npcap.
INTERFACE = os.environ.get("NIDS_INTERFACE", None)

# BPF filter (giong syntax tcpdump) - chi bat goi IP (tcp/udp/icmp) de giam tai
BPF_FILTER = "ip"

# ==================== FLOW 2 - Multi-Window Feature Extractor ====================
# Cac cua so thoi gian (giay) chay song song - moi cua so 1 thread rieng
WINDOW_SIZES = [1, 3, 5]

# Bao lau thi tinh lai feature 1 lan (giay)
EXTRACT_INTERVAL = 1.0

# Giu goi tin trong buffer toi da bao lau (phai >= max(WINDOW_SIZES))
MAX_BUFFER_SECONDS = max(WINDOW_SIZES) + 5

# Nguong Active/Idle kieu CICFlowMeter: IAT >= nguong nay -> tinh la khoang "Idle"
ACTIVE_IDLE_THRESHOLD = 1.0

# Flow khong co goi tin moi trong bao lau thi coi la "chet", don dep khoi bo nho
FLOW_TIMEOUT = 120

# ==================== FLOW 3 - Model Inference & Decision ====================
MODEL_DIR = os.environ.get("NIDS_MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "rf_final_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_final.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_list.pkl")

BENIGN_LABEL = "BENIGN"

# Cach ket hop du doan tu nhieu window: "majority" hoac "weighted"
VOTE_MODE = "weighted"

# Cua so dai hon -> tin cay hon voi tan cong keo dai (DDoS cham, port scan cham...)
WINDOW_WEIGHTS = {1: 1.0, 3: 1.2, 5: 1.5}

# Confidence toi thieu de 1 du doan tan cong duoc day di bao dong (loc bot false positive)
ALERT_MIN_CONFIDENCE = 0.55

# ==================== Kich thuoc Queue (chong tran bo nho) ====================
QUEUE1_MAXSIZE = 20000   # Raw packet queue (Flow1 -> Aggregator)
QUEUE2_MAXSIZE = 5000    # Feature vector queue (Flow2 -> Flow3)
ALERT_QUEUE_MAXSIZE = 2000  # Prediction queue (Flow3 -> Flow4)

# ==================== FLOW 4 - UI + Logging + Alert ====================
LOG_DIR = os.environ.get("NIDS_LOG_DIR", "./logs")
ALERT_LOG_FILE = os.path.join(LOG_DIR, "alerts.csv")
SYSTEM_LOG_FILE = os.path.join(LOG_DIR, "system.log")

# Thoi gian toi thieu giua 2 lan gui Telegram cho CUNG 1 flow (giay), tranh spam
ALERT_COOLDOWN_SECONDS = 5.0

# Telegram (tuy chon) - de trong neu khong dung, khong bao gio hardcode secret vao code
TELEGRAM_BOT_TOKEN = os.environ.get("NIDS_TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("NIDS_TELEGRAM_CHAT_ID", "")