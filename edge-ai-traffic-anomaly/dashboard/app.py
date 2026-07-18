"""
app.py — ET-SSL Local Dashboard (Streamlit)

Dashboard real-time hiển thị:
- Anomaly score chart (live stream giả lập)
- Alert feed (flows bị phân loại là anomaly)
- Model performance metrics
- Incremental learning status (μ_norm drift)

Chạy: streamlit run dashboard/app.py
"""

import sys
import os
import time
import json
import threading
from pathlib import Path
from datetime import datetime
from collections import deque
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.paths import get_model_dir, load_config
from pipeline.alert_manager import AlertManager
from pipeline.demo_stream import _make_flow
from pipeline.inference_runner import PipelineInferenceRunner
from pipeline.shared_state import load_pipeline_state

MODEL_DIR = str(get_model_dir())
CFG = load_config()
STATE_PATH = Path(CFG["logging"]["log_dir"]) / "pipeline_state.json"
ALERT_PATH = Path(CFG["dashboard"]["alert_log_path"])

# =====================================================================
# Page config
# =====================================================================
st.set_page_config(
    page_title="ET-SSL Edge AI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# Custom CSS
# =====================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif !important; }
    
    .main { background: #0a0e1a; }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a0e1a 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a2035 0%, #1e2840 100%);
        border: 1px solid rgba(100, 180, 255, 0.15);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(33, 150, 243, 0.2);
    }
    .metric-val { font-size: 2rem; font-weight: 700; margin: 8px 0; }
    .metric-label { font-size: 0.85rem; color: #8899aa; font-weight: 500; letter-spacing: 0.05em; }
    
    /* Alert items */
    .alert-item {
        background: linear-gradient(90deg, rgba(244,67,54,0.15) 0%, rgba(244,67,54,0.05) 100%);
        border-left: 3px solid #F44336;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.85rem;
    }
    .normal-item {
        background: linear-gradient(90deg, rgba(33,150,243,0.1) 0%, rgba(33,150,243,0.02) 100%);
        border-left: 3px solid #2196F3;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 3px 0;
        font-size: 0.8rem;
    }
    
    /* Header */
    .header-container {
        background: linear-gradient(135deg, #1a2035, #1e2840);
        border: 1px solid rgba(100, 180, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    
    /* Status pill */
    .status-online {
        display: inline-block;
        background: rgba(76, 175, 80, 0.2);
        border: 1px solid #4CAF50;
        color: #4CAF50;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Anomaly badge */
    .badge-anomaly {
        background: rgba(244,67,54,0.2);
        border: 1px solid #F44336;
        color: #FF6B6B;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .badge-normal {
        background: rgba(33,150,243,0.15);
        border: 1px solid rgba(33,150,243,0.4);
        color: #64B5F6;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
    }
    
    /* Streamlit overrides */
    .stMetric label { color: #8899aa !important; }
    .stMetric [data-testid="stMetricValue"] { color: #e8f0fe !important; }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a2035, #1e2840);
        border: 1px solid rgba(100, 180, 255, 0.12);
        border-radius: 12px;
        padding: 16px;
    }
    
    .sidebar-section {
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 14px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# State initialization
# =====================================================================
def _default_pipeline_mode() -> str:
    """Live Feed khi start.sh chạy demo/capture nền; Integrated khi chạy dashboard đơn."""
    if os.environ.get("ET_SSL_LIVE_FEED") == "1":
        return "live_feed"
    cfg = load_config()
    state_path = Path(cfg["logging"]["log_dir"]) / "pipeline_state.json"
    if state_path.exists():
        age = time.time() - state_path.stat().st_mtime
        if age < 120:
            return "live_feed"
    return "integrated"


def init_state():
    defaults = {
        "running": False,
        "scores": deque(maxlen=200),
        "is_anomaly": deque(maxlen=200),
        "timestamps": deque(maxlen=200),
        "alerts": deque(maxlen=50),
        "total_flows": 0,
        "total_anomalies": 0,
        "engine": None,
        "delta": 105.68,
        "alpha": 0.99,
        "mu_drift_history": deque(maxlen=100),
        "runner": None,
        "pipeline_mode": _default_pipeline_mode(),
        "mu_updates": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# =====================================================================
# Load engine (cached)
# =====================================================================
@st.cache_resource
def load_engine(model_dir, backend="onnx"):
    try:
        from model.inference import ETSSLInference
        engine = ETSSLInference(model_dir=model_dir, backend=backend)
        return engine, None
    except Exception as e:
        return None, str(e)


# Sidebar uses MODEL_DIR, CFG, STATE_PATH, ALERT_PATH from module top

# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**Pipeline Mode**")
    pipeline_mode = st.radio(
        "Nguồn traffic",
        ["integrated", "live_feed"],
        format_func=lambda x: "Integrated Pipeline" if x == "integrated" else "Live Feed (capture/demo)",
        index=0 if st.session_state.pipeline_mode == "integrated" else 1,
        help="Integrated: chạy full pipeline trong dashboard. Live Feed: đọc logs/pipeline_state.json",
    )
    st.session_state.pipeline_mode = pipeline_mode
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**Detection**")
    backend = st.selectbox("Inference Backend", ["onnx", "fp32", "int8"], index=0)
    kappa = st.slider("Sensitivity κ", 0.5, 3.0, float(CFG["detection"]["kappa"]), 0.1,
                      help="Ngưỡng thực = δ × κ")
    delta = st.slider("Base Threshold δ", 10.0, 500.0, st.session_state.delta, 5.0)
    st.session_state.delta = delta
    effective_delta = delta * kappa
    st.caption(f"Effective threshold: **{effective_delta:.1f}** (= δ × κ)")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**Incremental Learning**")
    alpha = st.slider("Decay α", 0.80, 0.999, st.session_state.alpha, 0.01,
                      help="α cao → cập nhật chậm (ổn định hơn)")
    update_interval = st.slider("Update Interval (N flows)", 10, 500, 100)
    st.session_state.alpha = alpha
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**Traffic Simulation**")
    anomaly_inject = st.slider("Anomaly injection rate (%)", 0, 50, 15)
    sim_speed = st.slider("Simulation speed (flows/sec)", 1, 20, 5)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ Start", use_container_width=True, type="primary"):
            st.session_state.running = True
    with col2:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.running = False

    if st.button("🔄 Reset Stats", use_container_width=True):
        st.session_state.scores.clear()
        st.session_state.is_anomaly.clear()
        st.session_state.timestamps.clear()
        st.session_state.alerts.clear()
        st.session_state.total_flows = 0
        st.session_state.total_anomalies = 0

    st.markdown("---")
    st.markdown("**Model Info**")
    st.caption(f"📁 `{Path(MODEL_DIR).name}`")
    st.caption("🔌 Encoder: 20→256→64")
    st.caption("📄 DOI: 10.1038/s41598-025-08568-0")


# =====================================================================
# Header
# =====================================================================
st.markdown("""
<div class="header-container">
    <div style="font-size:2.5rem">🛡️</div>
    <div>
        <h1 style="margin:0; font-size:1.8rem; color:#e8f0fe; font-weight:700;">
            ET-SSL Edge AI Dashboard
        </h1>
        <p style="margin:4px 0 0 0; color:#8899aa; font-size:0.9rem;">
            Anomaly Detection in Encrypted Network Traffic — Self-Supervised Learning
        </p>
    </div>
    <div style="margin-left:auto">
        <span class="status-online">● LIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# Load pipeline runner (integrated mode)
# =====================================================================
def get_runner(backend_name: str):
    if st.session_state.runner is None:
        runner = PipelineInferenceRunner.from_config(MODEL_DIR, backend=backend_name)
        st.session_state.runner = runner
    return st.session_state.runner


runner = None
if st.session_state.pipeline_mode == "integrated":
    runner = get_runner(backend)
    runner.engine.update_delta(delta)
    runner.kappa = kappa
    runner.learner.alpha = alpha
    runner.learner.update_interval = update_interval
    engine = runner.engine
    err = None
else:
    engine, err = load_engine(MODEL_DIR, backend)
    if engine:
        engine.update_delta(delta * kappa)

# =====================================================================
# KPI Metrics Row
# =====================================================================
flows = st.session_state.total_flows
anomalies = st.session_state.total_anomalies
anomaly_rate = (anomalies / flows * 100) if flows > 0 else 0.0

# Recent scores stats
scores_arr = np.array(list(st.session_state.scores)) if st.session_state.scores else np.array([])
avg_score = float(scores_arr.mean()) if len(scores_arr) > 0 else 0.0
max_score = float(scores_arr.max()) if len(scores_arr) > 0 else 0.0
threshold = delta * kappa

kpi_cols = st.columns(5)
kpis = [
    ("📊 Total Flows", f"{flows:,}", "#64B5F6"),
    ("🚨 Anomalies", f"{anomalies:,}", "#EF5350"),
    ("📈 Anomaly Rate", f"{anomaly_rate:.1f}%", "#FFA726"),
    ("⚡ Avg Score", f"{avg_score:.1f}", "#AB47BC"),
    ("🔺 Max Score", f"{max_score:.1f}", "#EF5350" if max_score > threshold else "#66BB6A"),
]

for col, (label, val, color) in zip(kpi_cols, kpis):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-val" style="color:{color}">{val}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================================
# Main charts
# =====================================================================
chart_col, alert_col = st.columns([3, 1])

with chart_col:
    st.markdown("### 📈 Anomaly Score — Real-time Stream")
    score_placeholder = st.empty()

    st.markdown("### 🔬 Score Distribution")
    dist_placeholder = st.empty()

with alert_col:
    st.markdown("### 🚨 Alert Feed")
    alert_placeholder = st.empty()

# =====================================================================
# Incremental learning status
# =====================================================================
st.markdown("---")
inc_col1, inc_col2, inc_col3 = st.columns(3)

with inc_col1:
    st.markdown("### 🔄 Incremental Learning")
    mu_drift_vals = np.array(list(st.session_state.mu_drift_history))
    if len(mu_drift_vals) > 1:
        st.line_chart(mu_drift_vals, use_container_width=True)
    else:
        st.info("μ_norm drift log sẽ hiện ở đây khi model bắt đầu học")

with inc_col2:
    st.markdown("### 📋 Pipeline Status")
    if runner and runner.learner:
        stats = runner.learner.stats
        st.json({
            "mode": st.session_state.pipeline_mode,
            "backend": backend,
            "delta (δ)": round(delta, 2),
            "kappa (κ)": kappa,
            "effective δ": round(delta * kappa, 2),
            "μ updates": stats["total_updates"],
            "normal seen": stats["total_normal_seen"],
            "anomaly seen": stats["total_anomaly_seen"],
            "buffer": stats["buffer_size"],
            "scaler": "✅" if runner.engine.scaler else "⚠️ Missing",
        })
    elif st.session_state.pipeline_mode == "live_feed":
        state = load_pipeline_state(STATE_PATH)
        st.json({
            "mode": "live_feed",
            "total_flows": state.get("total_flows", 0),
            "anomalies": state.get("total_anomalies", 0),
            "μ updates": state.get("mu_updates", 0),
            "last μ drift": state.get("last_mu_drift", 0),
        })
    elif engine:
        st.json({
            "backend": backend,
            "delta (δ)": round(engine.delta, 2),
            "embed_dim": engine.embed_dim,
            "input_dim": engine.input_dim,
        })

with inc_col3:
    st.markdown("### 📄 Paper Reference")
    st.markdown("""
    | Metric | Paper (RTX 3090) |
    |--------|-----------------|
    | Accuracy | **96.8%** |
    | TPR | **92.7%** |
    | FPR | **1.2%** |
    | Latency | **15–25 ms** |
    | Throughput | **~1900 fps** |
    """)
    st.caption("Sattar et al., *Scientific Reports* 2025")

# =====================================================================
# Pipeline loop — Integrated or Live Feed
# =====================================================================
def _sync_ui_from_state(state: dict):
    """Đồng bộ UI từ pipeline_state.json (live feed mode)."""
    if not state:
        return
    st.session_state.total_flows = state.get("total_flows", 0)
    st.session_state.total_anomalies = state.get("total_anomalies", 0)
    st.session_state.mu_updates = state.get("mu_updates", 0)
    st.session_state.scores = deque(state.get("scores", []), maxlen=200)
    st.session_state.is_anomaly = deque(state.get("is_anomaly", []), maxlen=200)
    st.session_state.timestamps = deque(state.get("timestamps", []), maxlen=200)
    hist = state.get("mu_drift_history", [])
    st.session_state.mu_drift_history = deque(hist, maxlen=100)


def _render_charts(scores_list, anomaly_list, threshold):
    import pandas as pd
    import plotly.graph_objects as go

    if not scores_list:
        return

    df = pd.DataFrame({"score": scores_list, "anomaly": anomaly_list})
    fig = go.Figure()
    colors = ["#EF5350" if a else "#42A5F5" for a in anomaly_list]
    fig.add_trace(go.Scatter(
        y=df["score"], mode="lines+markers", name="Anomaly Score",
        line=dict(color="#42A5F5", width=1.5),
        marker=dict(color=colors, size=5),
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#FF9800",
                  annotation_text=f"δ×κ={threshold:.0f}", annotation_position="top right")
    fig.update_layout(
        paper_bgcolor="#0d1526", plot_bgcolor="#0d1526",
        font=dict(color="#8899aa"), margin=dict(l=10, r=10, t=20, b=10),
        height=200, xaxis=dict(gridcolor="#1e2840"), yaxis=dict(gridcolor="#1e2840", title="S(x)"),
        showlegend=False,
    )
    score_placeholder.plotly_chart(fig, use_container_width=True)

    sc_arr = np.array(scores_list)
    ia_arr = np.array(anomaly_list)
    fig2 = go.Figure()
    if (~ia_arr).any():
        fig2.add_trace(go.Histogram(x=sc_arr[~ia_arr], name="Normal", marker_color="#42A5F5", opacity=0.7, nbinsx=30))
    if ia_arr.any():
        fig2.add_trace(go.Histogram(x=sc_arr[ia_arr], name="Anomaly", marker_color="#EF5350", opacity=0.7, nbinsx=30))
    fig2.add_vline(x=threshold, line_dash="dash", line_color="#FF9800")
    fig2.update_layout(
        paper_bgcolor="#0d1526", plot_bgcolor="#0d1526",
        font=dict(color="#8899aa"), barmode="overlay",
        margin=dict(l=10, r=10, t=20, b=10), height=180,
        xaxis=dict(gridcolor="#1e2840", title="Anomaly Score"),
        yaxis=dict(gridcolor="#1e2840", title="Count"),
    )
    dist_placeholder.plotly_chart(fig2, use_container_width=True)


def _render_alerts(alert_records, threshold):
    alert_html = ""
    if alert_records:
        for a in alert_records[:15]:
            ts = a.get("ts") or a.get("timestamp", "")[:19]
            score = a.get("score", 0)
            severity = "🔴" if score > threshold * 2 else "🟠"
            flow = a.get("flow_id", a.get("src", "?"))
            alert_html += f"""
            <div class="alert-item">
                {severity} <b>{ts}</b><br>
                {flow}<br>
                Score: <b>{score:.1f}</b> (δ×κ={threshold:.0f})
                <span class="badge-anomaly">ANOMALY</span>
            </div>
            """
    else:
        alert_html = "<div style='color:#8899aa; padding:20px; text-align:center;'>No alerts yet</div>"
    alert_placeholder.markdown(alert_html, unsafe_allow_html=True)


if st.session_state.running:
    if st.session_state.pipeline_mode == "live_feed":
        state = load_pipeline_state(STATE_PATH)
        _sync_ui_from_state(state)
        alerts_mgr = AlertManager(log_path=ALERT_PATH)
        alert_records = alerts_mgr.read_recent(15)
        sc = list(st.session_state.scores)
        ia = list(st.session_state.is_anomaly)
        if sc:
            _render_charts(sc, ia, threshold)
        _render_alerts(alert_records, threshold)
        time.sleep(CFG["dashboard"]["refresh_interval_sec"])
        st.rerun()

    elif runner:
        rng = np.random.default_rng()
        for i in range(sim_speed):
            is_anom = rng.random() < (anomaly_inject / 100)
            flow, features = _make_flow(st.session_state.total_flows + i, is_anom, rng)
            result = runner.process_flow(flow, features)

            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            score = result["score"]
            pred_anomaly = result["is_anomaly"]

            st.session_state.scores.append(score)
            st.session_state.is_anomaly.append(pred_anomaly)
            st.session_state.timestamps.append(ts)
            st.session_state.total_flows += 1

            if pred_anomaly:
                st.session_state.total_anomalies += 1
                st.session_state.alerts.appendleft({
                    "ts": ts, "score": score, "delta": threshold,
                    "flow_id": f"{flow.src_ip}:{flow.src_port}->{flow.dst_ip}:{flow.dst_port}",
                })

            if runner.learner and runner.learner.stats["total_updates"] > st.session_state.mu_updates:
                st.session_state.mu_updates = runner.learner.stats["total_updates"]
                hist = runner.state.mu_drift_history if runner.state else []
                st.session_state.mu_drift_history = deque(hist, maxlen=100)

        sc = list(st.session_state.scores)
        ia = list(st.session_state.is_anomaly)
        if sc:
            _render_charts(sc, ia, threshold)
        _render_alerts(list(st.session_state.alerts), threshold)
        time.sleep(1.0 / max(sim_speed, 1))
        st.rerun()

elif not st.session_state.running:
    score_placeholder.info("▶ Nhấn **Start** để bắt đầu simulation")
    dist_placeholder.info("Distribution chart sẽ hiển thị ở đây")
    alert_placeholder.info("Alert feed sẵn sàng")
