# ============================================================
# app.py  —  NIDS Streamlit Dashboard
# Run: streamlit run app.py
# pip install streamlit scikit-learn pandas numpy xgboost joblib plotly
# ============================================================

import streamlit as st
import joblib, json, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="NIDS Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800&display=swap');

* { font-family: 'Exo 2', sans-serif; }
code, .mono { font-family: 'Share Tech Mono', monospace; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0e1a;
    color: #c9d1e0;
}
[data-testid="stSidebar"] {
    background-color: #0d1220;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #c9d1e0 !important; }

h1, h2, h3 { font-family: 'Exo 2', sans-serif; font-weight: 800; }

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #0f1829 0%, #111e35 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 10px;
}
.metric-card .value { font-size: 2.2rem; font-weight: 800; font-family: 'Share Tech Mono'; }
.metric-card .label { font-size: 0.75rem; color: #6b7fa3; text-transform: uppercase; letter-spacing: 2px; margin-top: 4px; }

/* Attack badge */
.badge-attack {
    background: linear-gradient(90deg, #ff2d55, #ff6b35);
    color: white; padding: 6px 18px; border-radius: 20px;
    font-weight: 700; font-size: 1rem; display: inline-block;
    box-shadow: 0 0 20px rgba(255,45,85,0.4);
    animation: pulse 1.5s infinite;
}
.badge-normal {
    background: linear-gradient(90deg, #00d97e, #00b4d8);
    color: #0a0e1a; padding: 6px 18px; border-radius: 20px;
    font-weight: 700; font-size: 1rem; display: inline-block;
    box-shadow: 0 0 20px rgba(0,217,126,0.3);
}
@keyframes pulse {
    0%,100% { box-shadow: 0 0 20px rgba(255,45,85,0.4); }
    50%      { box-shadow: 0 0 35px rgba(255,45,85,0.8); }
}

/* Result box */
.result-box {
    border-radius: 16px; padding: 28px; margin: 20px 0;
    border: 1px solid;
}
.result-attack {
    background: rgba(255,45,85,0.08);
    border-color: rgba(255,45,85,0.4);
}
.result-normal {
    background: rgba(0,217,126,0.06);
    border-color: rgba(0,217,126,0.3);
}

/* Log table */
.log-entry {
    background: #0d1829;
    border: 1px solid #1e3050;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 6px 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
}
.log-attack { border-left: 3px solid #ff2d55; }
.log-normal { border-left: 3px solid #00d97e; }

/* Inputs */
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background-color: #0f1829 !important;
    border: 1px solid #1e3a5f !important;
    color: #c9d1e0 !important;
    border-radius: 8px !important;
}
.stSlider > div { padding: 0 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0d1220;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6b7fa3;
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #00d97e !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #0066ff, #0044cc);
    color: white; border: none; border-radius: 8px;
    font-weight: 700; padding: 10px 30px;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #0080ff, #0055ff);
    box-shadow: 0 0 20px rgba(0,102,255,0.4);
    transform: translateY(-1px);
}

/* Header banner */
.header-banner {
    background: linear-gradient(135deg, #0d1829 0%, #0f2040 50%, #0d1829 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.scan-line {
    height: 2px;
    background: linear-gradient(90deg, transparent, #00d97e, transparent);
    animation: scan 3s linear infinite;
    margin: 8px 0;
}
@keyframes scan {
    0%   { background-position: -100% 0; }
    100% { background-position: 200% 0; }
}

div[data-testid="stHorizontalBlock"] { gap: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    MODEL_DIR = "./nids_model"
    model   = joblib.load(f"{MODEL_DIR}/nids_model.pkl")
    scaler  = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    le_dict = joblib.load(f"{MODEL_DIR}/label_encoders.pkl")
    with open(f"{MODEL_DIR}/feature_names.json") as f:
        feature_names = json.load(f)
    with open(f"{MODEL_DIR}/metadata.json") as f:
        meta = json.load(f)
    return model, scaler, le_dict, feature_names, meta

model, scaler, le_dict, feature_names, meta = load_model()


# ── Predict function ──────────────────────────────────────────
def predict_traffic(raw_sample):
    df = pd.DataFrame([raw_sample])
    for col in meta["cat_columns"]:
        le = le_dict[col]
        df[col] = df[col].apply(
            lambda v: le.transform([v])[0] if v in le.classes_ else -1
        )
    df = df[feature_names]
    X  = scaler.transform(df)
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return {
        "prediction"   : meta["classes"][pred],
        "confidence"   : round(float(proba[pred]) * 100, 2),
        "normal_prob"  : round(float(proba[0]) * 100, 2),
        "attack_prob"  : round(float(proba[1]) * 100, 2),
    }


# ── Pre-built test samples ────────────────────────────────────
SAMPLES = {
    "— Select a preset —": None,

    "🟢 Normal HTTP Traffic": {
        "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
        "src_bytes": 215, "dst_bytes": 45076, "land": 0, "wrong_fragment": 0,
        "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 1,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 1, "srv_count": 1, "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 255,
        "dst_host_srv_count": 255, "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.0,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0
    },
    "🟢 Normal FTP Session": {
        "duration": 15, "protocol_type": "tcp", "service": "ftp", "flag": "SF",
        "src_bytes": 2000, "dst_bytes": 3000, "land": 0, "wrong_fragment": 0,
        "urgent": 0, "hot": 2, "num_failed_logins": 0, "logged_in": 1,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 1, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 3, "srv_count": 3, "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 50,
        "dst_host_srv_count": 50, "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.04,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0
    },
    "🟢 Normal DNS (UDP)": {
        "duration": 0, "protocol_type": "udp", "service": "domain", "flag": "SF",
        "src_bytes": 38, "dst_bytes": 56, "land": 0, "wrong_fragment": 0,
        "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 0,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 2, "srv_count": 2, "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 30,
        "dst_host_srv_count": 30, "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.03,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0
    },
    "🟢 Normal SMTP (Email)": {
        "duration": 5, "protocol_type": "tcp", "service": "smtp", "flag": "SF",
        "src_bytes": 750, "dst_bytes": 300, "land": 0, "wrong_fragment": 0,
        "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 1,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 5, "srv_count": 5, "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 20,
        "dst_host_srv_count": 20, "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.05,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0
    },
    "🟢 Normal SSH Session": {
        "duration": 30, "protocol_type": "tcp", "service": "ssh", "flag": "SF",
        "src_bytes": 3000, "dst_bytes": 5000, "land": 0, "wrong_fragment": 0,
        "urgent": 0, "hot": 1, "num_failed_logins": 0, "logged_in": 1,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 1, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 2, "srv_count": 2, "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 10,
        "dst_host_srv_count": 10, "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.1,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0
    },
    "🔴 DoS SYN Flood": {
        "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "S0",
        "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0,
        "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 0,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 511, "srv_count": 511, "serror_rate": 1.0, "srv_serror_rate": 1.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 255,
        "dst_host_srv_count": 255, "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 1.0,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 1.0,
        "dst_host_srv_serror_rate": 1.0, "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0
    },
    "🔴 DoS ICMP Flood": {
        "duration": 0, "protocol_type": "icmp", "service": "eco_i", "flag": "SF",
        "src_bytes": 1032, "dst_bytes": 0, "land": 0, "wrong_fragment": 0,
        "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 0,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 511, "srv_count": 511, "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 255,
        "dst_host_srv_count": 255, "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 1.0,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0
    },
    "🔴 Land Attack": {
        "duration": 0, "protocol_type": "tcp", "service": "telnet", "flag": "S0",
        "src_bytes": 0, "dst_bytes": 0, "land": 1, "wrong_fragment": 0,
        "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 0,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 1, "srv_count": 1, "serror_rate": 1.0, "srv_serror_rate": 1.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 255,
        "dst_host_srv_count": 255, "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 1.0,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 1.0,
        "dst_host_srv_serror_rate": 1.0, "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0
    },
    "🔴 Port Scan (Probe)": {
        "duration": 0, "protocol_type": "tcp", "service": "private", "flag": "REJ",
        "src_bytes": 0, "dst_bytes": 0, "land": 0, "wrong_fragment": 0,
        "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 0,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 255, "srv_count": 1, "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 1.0, "srv_rerror_rate": 1.0, "same_srv_rate": 0.0,
        "diff_srv_rate": 1.0, "srv_diff_host_rate": 0.0, "dst_host_count": 255,
        "dst_host_srv_count": 1, "dst_host_same_srv_rate": 0.0,
        "dst_host_diff_srv_rate": 1.0, "dst_host_same_src_port_rate": 0.0,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 1.0,
        "dst_host_srv_rerror_rate": 1.0
    },
    "🔴 R2L Brute Force": {
        "duration": 2, "protocol_type": "tcp", "service": "telnet", "flag": "SF",
        "src_bytes": 1200, "dst_bytes": 800, "land": 0, "wrong_fragment": 0,
        "urgent": 0, "hot": 0, "num_failed_logins": 5, "logged_in": 0,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 1, "srv_count": 1, "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 10,
        "dst_host_srv_count": 10, "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.1,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0
    },
    "🔴 U2R Privilege Escalation": {
        "duration": 0, "protocol_type": "tcp", "service": "telnet", "flag": "SF",
        "src_bytes": 1408, "dst_bytes": 1593, "land": 0, "wrong_fragment": 0,
        "urgent": 0, "hot": 8, "num_failed_logins": 0, "logged_in": 1,
        "num_compromised": 1, "root_shell": 1, "su_attempted": 1, "num_root": 4,
        "num_file_creations": 0, "num_shells": 1, "num_access_files": 1,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 1, "srv_count": 1, "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 1,
        "dst_host_srv_count": 1, "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 1.0,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0
    },
}

# ── Session state ─────────────────────────────────────────────
if "log" not in st.session_state:
    st.session_state.log = []
if "stats" not in st.session_state:
    st.session_state.stats = {"total": 0, "attacks": 0, "normal": 0}


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-banner">
  <div style="font-size:2.8rem">🛡️</div>
  <div>
    <div style="font-size:1.6rem;font-weight:800;color:#e2e8f0;letter-spacing:1px">
      Network Intrusion Detection System
    </div>
    <div style="font-size:0.8rem;color:#6b7fa3;letter-spacing:3px;text-transform:uppercase">
      ML-Powered · XGBoost · NSL-KDD Dataset
    </div>
    <div class="scan-line"></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Top metrics ───────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="value" style="color:#00d97e">{meta['accuracy']*100:.1f}%</div>
        <div class="label">Model Accuracy</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="value" style="color:#00b4d8">{meta['f1_score']*100:.1f}%</div>
        <div class="label">F1 Score</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="value" style="color:#ff9f1c">{st.session_state.stats['total']}</div>
        <div class="label">Total Scanned</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="value" style="color:#ff2d55">{st.session_state.stats['attacks']}</div>
        <div class="label">Attacks Detected</div></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔍  Single Scan", "⚡  Batch Test", "📊  Analytics"])


# ─────────────────────────────────────────────────────────────
# TAB 1 — SINGLE SCAN
# ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Analyze a Network Connection")

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### Quick Load Preset")
        preset = st.selectbox("Choose a preset traffic sample", list(SAMPLES.keys()), label_visibility="collapsed")

        # Defaults
        defaults = SAMPLES[preset] if SAMPLES[preset] else SAMPLES["🟢 Normal HTTP Traffic"]

        st.markdown("#### Connection Details")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            protocol = st.selectbox("Protocol", ["tcp", "udp", "icmp"],
                index=["tcp","udp","icmp"].index(defaults.get("protocol_type","tcp")))
        with col_b:
            services = ["http","ftp","smtp","ssh","telnet","domain","private","ftp_data",
                        "eco_i","other"]
            svc_val = defaults.get("service","http")
            svc_idx = services.index(svc_val) if svc_val in services else 0
            service = st.selectbox("Service", services, index=svc_idx)
        with col_c:
            flags = ["SF","S0","REJ","RSTO","RSTR","SH","OTH","S1","S2","S3"]
            flag_val = defaults.get("flag","SF")
            flag_idx = flags.index(flag_val) if flag_val in flags else 0
            flag = st.selectbox("Flag", flags, index=flag_idx)

        col_d, col_e = st.columns(2)
        with col_d:
            src_bytes = st.number_input("Src Bytes", value=int(defaults.get("src_bytes",0)), min_value=0)
            duration  = st.number_input("Duration (s)", value=int(defaults.get("duration",0)), min_value=0)
            count     = st.number_input("Count", value=int(defaults.get("count",1)), min_value=0)
        with col_e:
            dst_bytes = st.number_input("Dst Bytes", value=int(defaults.get("dst_bytes",0)), min_value=0)
            logged_in = st.selectbox("Logged In", [0,1], index=int(defaults.get("logged_in",0)))
            serror_rate = st.slider("SError Rate", 0.0, 1.0, float(defaults.get("serror_rate",0.0)), 0.01)

        st.markdown("#### Advanced Flags")
        col_f, col_g, col_h = st.columns(3)
        with col_f:
            land       = st.selectbox("Land", [0,1], index=int(defaults.get("land",0)))
            root_shell = st.selectbox("Root Shell", [0,1], index=int(defaults.get("root_shell",0)))
        with col_g:
            num_failed = st.number_input("Failed Logins", value=int(defaults.get("num_failed_logins",0)), min_value=0)
            su_attempt = st.selectbox("SU Attempted", [0,1], index=int(defaults.get("su_attempted",0)))
        with col_h:
            num_shells = st.number_input("Num Shells", value=int(defaults.get("num_shells",0)), min_value=0)
            num_root   = st.number_input("Num Root", value=int(defaults.get("num_root",0)), min_value=0)

        scan_btn = st.button("🔍  ANALYZE TRAFFIC", use_container_width=True)

    with right:
        st.markdown("#### Detection Result")

        if scan_btn:
            sample = dict(defaults)
            sample.update({
                "protocol_type": protocol, "service": service, "flag": flag,
                "src_bytes": src_bytes, "dst_bytes": dst_bytes,
                "duration": duration, "count": count, "logged_in": logged_in,
                "serror_rate": serror_rate, "land": land,
                "root_shell": root_shell, "num_failed_logins": num_failed,
                "su_attempted": su_attempt, "num_shells": num_shells,
                "num_root": num_root,
            })

            result = predict_traffic(sample)
            is_attack = result["prediction"] == "Attack"

            # Update stats
            st.session_state.stats["total"] += 1
            if is_attack:
                st.session_state.stats["attacks"] += 1
            else:
                st.session_state.stats["normal"] += 1

            # Log entry
            st.session_state.log.append({
                "time"      : datetime.now().strftime("%H:%M:%S"),
                "protocol"  : protocol,
                "service"   : service,
                "flag"      : flag,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
            })

            box_cls = "result-attack" if is_attack else "result-normal"
            badge   = f'<span class="badge-attack">⚠ ATTACK DETECTED</span>' if is_attack \
                      else f'<span class="badge-normal">✓ NORMAL TRAFFIC</span>'
            color   = "#ff2d55" if is_attack else "#00d97e"

            st.markdown(f"""
            <div class="result-box {box_cls}">
              <div style="margin-bottom:12px">{badge}</div>
              <div style="font-size:3rem;font-weight:800;color:{color};
                          font-family:'Share Tech Mono';margin:8px 0">
                {result['confidence']}%
              </div>
              <div style="color:#6b7fa3;font-size:0.8rem;letter-spacing:2px">CONFIDENCE</div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["attack_prob"],
                title={"text": "Attack Probability", "font": {"color": "#c9d1e0", "size": 14}},
                gauge={
                    "axis"      : {"range": [0,100], "tickcolor": "#6b7fa3"},
                    "bar"       : {"color": "#ff2d55" if is_attack else "#00d97e"},
                    "bgcolor"   : "#0d1829",
                    "bordercolor": "#1e3a5f",
                    "steps"     : [
                        {"range": [0, 30],  "color": "rgba(0,217,126,0.1)"},
                        {"range": [30, 70], "color": "rgba(255,159,28,0.1)"},
                        {"range": [70,100], "color": "rgba(255,45,85,0.1)"},
                    ],
                    "threshold" : {"value": 50, "line": {"color": "#fff", "width": 2}, "thickness": 0.75}
                },
                number={"suffix": "%", "font": {"color": "#c9d1e0", "size": 28}}
            ))
            fig.update_layout(
                paper_bgcolor="#0a0e1a", font_color="#c9d1e0",
                height=260, margin=dict(t=40, b=10, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Prob bar
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=["Normal", "Attack"],
                y=[result["normal_prob"], result["attack_prob"]],
                marker_color=["#00d97e", "#ff2d55"],
                text=[f"{result['normal_prob']}%", f"{result['attack_prob']}%"],
                textposition="outside",
                textfont=dict(color="#c9d1e0", family="Share Tech Mono")
            ))
            fig2.update_layout(
                paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1829",
                font_color="#c9d1e0", height=200,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#1e3050", range=[0,115]),
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

        else:
            st.markdown("""
            <div style="text-align:center;padding:80px 20px;color:#3a4f6b;
                        border:2px dashed #1e3050;border-radius:16px;margin-top:20px">
                <div style="font-size:3rem;margin-bottom:16px">🔍</div>
                <div style="font-size:1rem;font-weight:600">Configure parameters and click<br>ANALYZE TRAFFIC</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TAB 2 — BATCH TEST
# ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Run All Preset Samples")

    if st.button("⚡  RUN FULL BATCH TEST", use_container_width=False):
        batch_results = []
        progress = st.progress(0)
        samples_to_run = {k: v for k, v in SAMPLES.items() if v is not None}

        for i, (name, sample) in enumerate(samples_to_run.items()):
            result = predict_traffic(sample)
            is_attack = result["prediction"] == "Attack"
            expected  = "Normal" if "🟢" in name else "Attack"
            correct   = result["prediction"] == expected
            batch_results.append({
                "Traffic Type"  : name,
                "Expected"      : expected,
                "Predicted"     : result["prediction"],
                "Confidence"    : f"{result['confidence']}%",
                "Correct"       : "✅" if correct else "❌"
            })
            st.session_state.stats["total"] += 1
            if is_attack:
                st.session_state.stats["attacks"] += 1
            else:
                st.session_state.stats["normal"] += 1

            progress.progress((i+1) / len(samples_to_run))

        df_results = pd.DataFrame(batch_results)
        correct_count = df_results["Correct"].value_counts().get("✅", 0)
        total_count   = len(df_results)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="value" style="color:#00d97e">{correct_count}/{total_count}</div>
                <div class="label">Correct</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="value" style="color:#00b4d8">{correct_count/total_count*100:.1f}%</div>
                <div class="label">Batch Accuracy</div></div>""", unsafe_allow_html=True)
        with c3:
            wrong = total_count - correct_count
            st.markdown(f"""<div class="metric-card">
                <div class="value" style="color:#ff2d55">{wrong}</div>
                <div class="label">Misclassified</div></div>""", unsafe_allow_html=True)

        st.markdown("#### Detailed Results")
        st.dataframe(
            df_results,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Traffic Type"  : st.column_config.TextColumn(width="large"),
                "Confidence"    : st.column_config.TextColumn(width="small"),
            }
        )

        # Pie chart
        attack_count = df_results[df_results["Predicted"]=="Attack"].shape[0]
        normal_count = df_results[df_results["Predicted"]=="Normal"].shape[0]
        fig_pie = go.Figure(go.Pie(
            labels=["Normal", "Attack"],
            values=[normal_count, attack_count],
            hole=0.6,
            marker=dict(colors=["#00d97e","#ff2d55"],
                        line=dict(color="#0a0e1a", width=3))
        ))
        fig_pie.update_layout(
            paper_bgcolor="#0a0e1a", font_color="#c9d1e0",
            height=320, showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            title=dict(text="Prediction Distribution", font=dict(color="#c9d1e0"))
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#3a4f6b;
                    border:2px dashed #1e3050;border-radius:16px">
            <div style="font-size:2.5rem;margin-bottom:12px">⚡</div>
            <div>Click the button above to run all 11 preset samples at once</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TAB 3 — ANALYTICS
# ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Session Analytics")

    if st.session_state.log:
        log_df = pd.DataFrame(st.session_state.log)

        col1, col2 = st.columns(2)

        with col1:
            # Attacks over time
            fig_t = px.line(
                log_df.reset_index(), x="index", y="confidence",
                color="prediction",
                color_discrete_map={"Normal": "#00d97e", "Attack": "#ff2d55"},
                title="Confidence Over Time",
                labels={"index": "Scan #", "confidence": "Confidence (%)"}
            )
            fig_t.update_layout(
                paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1829",
                font_color="#c9d1e0", height=300,
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor="#1e3050"),
                yaxis=dict(gridcolor="#1e3050")
            )
            st.plotly_chart(fig_t, use_container_width=True)

        with col2:
            # Protocol distribution
            proto_counts = log_df["protocol"].value_counts()
            fig_proto = go.Figure(go.Bar(
                x=proto_counts.index,
                y=proto_counts.values,
                marker_color=["#0066ff","#00d97e","#ff9f1c"],
                text=proto_counts.values,
                textposition="outside"
            ))
            fig_proto.update_layout(
                title="Protocol Distribution",
                paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1829",
                font_color="#c9d1e0", height=300,
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="#1e3050")
            )
            st.plotly_chart(fig_proto, use_container_width=True)

        st.markdown("#### Scan Log")
        for entry in reversed(st.session_state.log[-20:]):
            cls = "log-attack" if entry["prediction"] == "Attack" else "log-normal"
            icon = "🔴" if entry["prediction"] == "Attack" else "🟢"
            st.markdown(f"""
            <div class="log-entry {cls}">
                {icon} [{entry['time']}] &nbsp;
                <b>{entry['prediction']}</b> &nbsp;·&nbsp;
                {entry['protocol'].upper()} / {entry['service']} / {entry['flag']} &nbsp;·&nbsp;
                Confidence: <b>{entry['confidence']}%</b>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Log"):
            st.session_state.log   = []
            st.session_state.stats = {"total": 0, "attacks": 0, "normal": 0}
            st.rerun()
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#3a4f6b;
                    border:2px dashed #1e3050;border-radius:16px">
            <div style="font-size:2.5rem;margin-bottom:12px">📊</div>
            <div>No scans yet. Run some traffic analyses to see analytics here.</div>
        </div>
        """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ Model Info")
    st.markdown(f"""
    <div style="background:#0f1829;border:1px solid #1e3a5f;border-radius:10px;padding:16px;font-size:0.85rem">
        <div style="margin:6px 0"><span style="color:#6b7fa3">Model</span><br>
            <b style="color:#00d97e">{meta['model_type']}</b></div>
        <div style="margin:6px 0"><span style="color:#6b7fa3">Accuracy</span><br>
            <b style="color:#00b4d8">{meta['accuracy']*100:.2f}%</b></div>
        <div style="margin:6px 0"><span style="color:#6b7fa3">F1 Score</span><br>
            <b style="color:#00b4d8">{meta['f1_score']*100:.2f}%</b></div>
        <div style="margin:6px 0"><span style="color:#6b7fa3">Dataset</span><br>
            <b>NSL-KDD</b></div>
        <div style="margin:6px 0"><span style="color:#6b7fa3">Features</span><br>
            <b>{meta['n_features']}</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Session Stats")
    total   = st.session_state.stats["total"]
    attacks = st.session_state.stats["attacks"]
    normal  = st.session_state.stats["normal"]
    rate    = f"{attacks/total*100:.1f}%" if total > 0 else "—"

    st.markdown(f"""
    <div style="background:#0f1829;border:1px solid #1e3a5f;border-radius:10px;padding:16px;font-size:0.85rem">
        <div style="margin:6px 0"><span style="color:#6b7fa3">Total Scans</span><br>
            <b style="color:#ff9f1c">{total}</b></div>
        <div style="margin:6px 0"><span style="color:#6b7fa3">Normal</span><br>
            <b style="color:#00d97e">{normal}</b></div>
        <div style="margin:6px 0"><span style="color:#6b7fa3">Attacks</span><br>
            <b style="color:#ff2d55">{attacks}</b></div>
        <div style="margin:6px 0"><span style="color:#6b7fa3">Attack Rate</span><br>
            <b style="color:#ff9f1c">{rate}</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ℹ️ Attack Categories")
    st.markdown("""
    <div style="font-size:0.8rem;color:#6b7fa3;line-height:1.8">
        🔴 <b style="color:#c9d1e0">DoS</b> — Denial of Service<br>
        🔴 <b style="color:#c9d1e0">Probe</b> — Network Scanning<br>
        🔴 <b style="color:#c9d1e0">R2L</b> — Remote to Local<br>
        🔴 <b style="color:#c9d1e0">U2R</b> — User to Root<br>
        🟢 <b style="color:#c9d1e0">Normal</b> — Benign Traffic
    </div>
    """, unsafe_allow_html=True)
