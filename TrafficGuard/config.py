import os

LOG_LEVEL = os.environ.get("NIDS_LOG_LEVEL", "INFO").upper()

INTERFACE = os.environ.get("NIDS_INTERFACE", None)

BPF_FILTER = "ip"

WINDOW_SIZES = [1, 3, 5]

EXTRACT_INTERVAL = 1.0

MAX_BUFFER_SECONDS = max(WINDOW_SIZES) + 5

ACTIVE_IDLE_THRESHOLD = 1.0

FLOW_TIMEOUT = 120

MODEL_DIR = os.environ.get("NIDS_MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "rf_final_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_final.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
FEATURE_LIST_PATH = os.path.join(MODEL_DIR, "feature_list.pkl")

BENIGN_LABEL = "BENIGN"

VOTE_MODE = "weighted"

WINDOW_WEIGHTS = {1: 1.0, 3: 1.2, 5: 1.5}

ALERT_MIN_CONFIDENCE = 0.55

CONFIRM_STREAK = 2            # <-- THEM MOI: phai lien tiep bi nghi la attack bao nhieu lan moi bao dong that
STALE_WINDOW_SECONDS = 15.0   # <-- THEM MOI: window nao cu qua (qua so giay nay) thi bi loai khoi vote

QUEUE1_MAXSIZE = 20000   # Raw packet queue (Flow1 -> Aggregator)
QUEUE2_MAXSIZE = 5000    # Feature vector queue (Flow2 -> Flow3)
ALERT_QUEUE_MAXSIZE = 2000  # Prediction queue (Flow3 -> Flow4)

LOG_DIR = os.environ.get("NIDS_LOG_DIR", "./logs")
ALERT_LOG_FILE = os.path.join(LOG_DIR, "alerts.csv")
SYSTEM_LOG_FILE = os.path.join(LOG_DIR, "system.log")

ALERT_COOLDOWN_SECONDS = 5.0

TELEGRAM_BOT_TOKEN = os.environ.get("NIDS_TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("NIDS_TELEGRAM_CHAT_ID", "")