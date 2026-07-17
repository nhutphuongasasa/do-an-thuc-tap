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
import time
import json
import threading
from pathlib import Path
from datetime import datetime
from collections import deque
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

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
        "demo_mode": True,
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


MODEL_DIR = str(
    Path(__file__).parent.parent.parent / "TrafficGuard/models/edge_ai-20260716T101644Z-1-001/edge_ai"
)

# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    backend = st.selectbox("Inference Backend", ["onnx", "fp32", "int8"], index=0)
    delta = st.slider("Anomaly Threshold δ", 10.0, 500.0, st.session_state.delta, 5.0)
    st.session_state.delta = delta
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**Incremental Learning**")
    alpha = st.slider("Decay α", 0.80, 0.999, st.session_state.alpha, 0.01,
                      help="α cao → cập nhật chậm (ổn định hơn)")
    update_interval = st.slider("Update Interval (N flows)", 10, 500, 100)
    st.session_state.alpha = alpha
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("**Demo Settings**")
    demo_mode = st.checkbox("Demo Mode (synthetic data)", value=True,
                            help="Tạo traffic ngẫu nhiên để demo")
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
# Load model
# =====================================================================
engine, err = load_engine(MODEL_DIR, backend)
if engine:
    engine.update_delta(delta)

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

kpi_cols = st.columns(5)
kpis = [
    ("📊 Total Flows", f"{flows:,}", "#64B5F6"),
    ("🚨 Anomalies", f"{anomalies:,}", "#EF5350"),
    ("📈 Anomaly Rate", f"{anomaly_rate:.1f}%", "#FFA726"),
    ("⚡ Avg Score", f"{avg_score:.1f}", "#AB47BC"),
    ("🔺 Max Score", f"{max_score:.1f}", "#EF5350" if max_score > delta else "#66BB6A"),
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
    st.markdown("### 📋 Model Config")
    if engine:
        st.json({
            "backend": backend,
            "delta (δ)": round(engine.delta, 2),
            "embed_dim": engine.embed_dim,
            "input_dim": engine.input_dim,
            "scaler": "✅" if engine.scaler else "⚠️ Missing",
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
# Simulation loop
# =====================================================================
def _generate_demo_flow(anomaly_ratio: float, rng):
    """Generate 1 synthetic flow feature vector."""
    is_anom = rng.random() < (anomaly_ratio / 100)
    if is_anom:
        x = rng.normal(3.0, 1.5, 20).astype(np.float32)
    else:
        x = rng.normal(0.0, 1.0, 20).astype(np.float32)
    return x, is_anom


if st.session_state.running:
    rng = np.random.default_rng()

    for _ in range(sim_speed):
        x, ground_truth = _generate_demo_flow(anomaly_inject, rng)

        if engine:
            result = engine.predict(x)
            score = result["score"]
            pred_anomaly = result["is_anomaly"]
        else:
            # Fallback nếu không load được model
            score = float(np.sum(x**2)) + rng.normal(0, 5)
            pred_anomaly = score > delta

        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        st.session_state.scores.append(score)
        st.session_state.is_anomaly.append(pred_anomaly)
        st.session_state.timestamps.append(ts)
        st.session_state.total_flows += 1

        if pred_anomaly:
            st.session_state.total_anomalies += 1
            st.session_state.alerts.appendleft({
                "ts": ts,
                "score": score,
                "delta": delta,
            })

    # Update charts
    sc = list(st.session_state.scores)
    ia = list(st.session_state.is_anomaly)

    if sc:
        import pandas as pd
        import plotly.graph_objects as go

        df = pd.DataFrame({
            "score": sc,
            "anomaly": ia,
        })

        # Score time series
        fig = go.Figure()
        colors = ["#EF5350" if a else "#42A5F5" for a in ia]
        fig.add_trace(go.Scatter(
            y=df["score"], mode="lines+markers",
            name="Anomaly Score",
            line=dict(color="#42A5F5", width=1.5),
            marker=dict(color=colors, size=5),
        ))
        fig.add_hline(y=delta, line_dash="dash", line_color="#FF9800",
                      annotation_text=f"δ={delta:.0f}", annotation_position="top right")
        fig.update_layout(
            paper_bgcolor="#0d1526", plot_bgcolor="#0d1526",
            font=dict(color="#8899aa"),
            margin=dict(l=10, r=10, t=20, b=10),
            height=200,
            xaxis=dict(gridcolor="#1e2840", title=""),
            yaxis=dict(gridcolor="#1e2840", title="S(x)"),
            showlegend=False,
        )
        score_placeholder.plotly_chart(fig, use_container_width=True)

        # Distribution
        sc_arr = np.array(sc)
        ia_arr = np.array(ia)
        fig2 = go.Figure()
        if (ia_arr == False).any():
            fig2.add_trace(go.Histogram(
                x=sc_arr[ia_arr == False], name="Normal",
                marker_color="#42A5F5", opacity=0.7,
                nbinsx=30,
            ))
        if ia_arr.any():
            fig2.add_trace(go.Histogram(
                x=sc_arr[ia_arr], name="Anomaly",
                marker_color="#EF5350", opacity=0.7,
                nbinsx=30,
            ))
        fig2.add_vline(x=delta, line_dash="dash", line_color="#FF9800")
        fig2.update_layout(
            paper_bgcolor="#0d1526", plot_bgcolor="#0d1526",
            font=dict(color="#8899aa"),
            barmode="overlay",
            margin=dict(l=10, r=10, t=20, b=10),
            height=180,
            xaxis=dict(gridcolor="#1e2840", title="Anomaly Score"),
            yaxis=dict(gridcolor="#1e2840", title="Count"),
        )
        dist_placeholder.plotly_chart(fig2, use_container_width=True)

    # Alert feed
    alerts = list(st.session_state.alerts)
    alert_html = ""
    if alerts:
        for a in alerts[:15]:
            severity = "🔴" if a["score"] > delta * 2 else "🟠"
            alert_html += f"""
            <div class="alert-item">
                {severity} <b>{a['ts']}</b><br>
                Score: <b>{a['score']:.1f}</b> (δ={a['delta']:.0f})
                <span class="badge-anomaly">ANOMALY</span>
            </div>
            """
    else:
        alert_html = "<div style='color:#8899aa; padding:20px; text-align:center;'>No alerts yet</div>"

    alert_placeholder.markdown(alert_html, unsafe_allow_html=True)

    time.sleep(1.0 / max(sim_speed, 1))
    st.rerun()

elif not st.session_state.running:
    score_placeholder.info("▶ Nhấn **Start** để bắt đầu simulation")
    dist_placeholder.info("Distribution chart sẽ hiển thị ở đây")
    alert_placeholder.info("Alert feed sẵn sàng")
