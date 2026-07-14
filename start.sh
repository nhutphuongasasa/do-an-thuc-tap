#!/bin/sh

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"
MAIN_SCRIPT="$SCRIPT_DIR/darktrace-nids/main.py"

echo "=========================================================="
echo " NIDS/NDR Pipeline Startup"
echo "=========================================================="

# ── Check venv ─────────────────────────────────────────────────
if [ ! -f "$PYTHON_BIN" ]; then
    echo "Error: Virtual environment not found at $SCRIPT_DIR/venv"
    echo "Please create it first:  python3 -m venv venv && pip install -r darktrace-nids/requirements.txt"
    exit 1
fi

# ── Auto-set CAP_NET_RAW if not already present ────────────────
# This allows Scapy to capture raw packets without running as root.
if command -v getcap >/dev/null 2>&1; then
    HAS_CAP=$(getcap "$PYTHON_BIN" 2>/dev/null | grep "cap_net_raw")
fi

if [ -z "$HAS_CAP" ]; then
    echo "Setting packet capture capabilities on Python (requires sudo once)..."
    sudo setcap cap_net_raw,cap_net_admin=eip "$PYTHON_BIN"

    if [ $? -ne 0 ]; then
        echo ""
        echo "Warning: Could not set capabilities. Falling back to sudo execution."
        echo "=========================================================="
        exec sudo "$PYTHON_BIN" "$MAIN_SCRIPT"
    fi

    echo "Capabilities set. You will not be asked again."
fi

echo "Starting NIDS pipeline..."
echo "=========================================================="
exec "$PYTHON_BIN" "$MAIN_SCRIPT"
