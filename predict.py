# ============================================================
# predict.py  —  NIDS Inference Script
# Run: python predict.py
# Requirements: pip install scikit-learn pandas numpy xgboost joblib
# ============================================================

import joblib, json
import numpy as np
import pandas as pd

# ── Load artifacts ────────────────────────────────────────────
MODEL_DIR = "./nids_model"

model     = joblib.load(f"C:/Users/KIIT0001/Desktop/NIDS/nids_model/nids_model.pkl")
scaler    = joblib.load(f"C:/Users/KIIT0001/Desktop/NIDS/nids_model/scaler.pkl")
le_dict   = joblib.load(f"C:/Users/KIIT0001/Desktop/NIDS/nids_model/label_encoders.pkl")

with open(f"C:/Users/KIIT0001/Desktop/NIDS/nids_model/feature_names.json") as f:
    feature_names = json.load(f)

with open(f"C:/Users/KIIT0001/Desktop/NIDS/nids_model/metadata.json") as f:
    meta = json.load(f)

print(f"\n✅ Loaded {meta['model_type']}  |  Accuracy: {meta['accuracy']}  |  F1: {meta['f1_score']}")

# ── Prediction function ───────────────────────────────────────
def predict_traffic(raw_sample: dict) -> dict:
    """
    raw_sample: dict with keys matching feature_names.
    Categorical fields (protocol_type, service, flag) should be raw strings.
    """
    df = pd.DataFrame([raw_sample])

    # Encode categorical columns
    for col in meta["cat_columns"]:
        le = le_dict[col]
        df[col] = df[col].apply(
            lambda v: le.transform([v])[0] if v in le.classes_ else -1
        )

    # Reorder & scale
    df = df[feature_names]
    X  = scaler.transform(df)

    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    return {
        "prediction"   : meta["classes"][pred],
        "confidence"   : round(float(proba[pred]) * 100, 2),
        "probabilities": {
            "Normal": round(float(proba[0]) * 100, 2),
            "Attack": round(float(proba[1]) * 100, 2)
        }
    }


# ── Test Samples ──────────────────────────────────────────────
test_samples = {

    # ── NORMAL TRAFFIC ────────────────────────────────────────

    "Normal HTTP Traffic": {
        "duration": 0, "protocol_type": "tcp", "service": "http",
        "flag": "SF", "src_bytes": 215, "dst_bytes": 45076,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
        "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 1, "srv_count": 1, "serror_rate": 0.0,
        "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 255, "dst_host_srv_count": 255,
        "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.0, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    },

    "Normal FTP Session": {
        "duration": 15, "protocol_type": "tcp", "service": "ftp",
        "flag": "SF", "src_bytes": 2000, "dst_bytes": 3000,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 2,
        "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 1, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 3, "srv_count": 3, "serror_rate": 0.0,
        "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 50, "dst_host_srv_count": 50,
        "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.04, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    },

    "Normal DNS (UDP)": {
        "duration": 0, "protocol_type": "udp", "service": "domain",
        "flag": "SF", "src_bytes": 38, "dst_bytes": 56,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
        "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 2, "srv_count": 2, "serror_rate": 0.0,
        "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 30, "dst_host_srv_count": 30,
        "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.03, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    },

    "Normal SMTP (Email)": {
        "duration": 5, "protocol_type": "tcp", "service": "smtp",
        "flag": "SF", "src_bytes": 750, "dst_bytes": 300,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
        "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 5, "srv_count": 5, "serror_rate": 0.0,
        "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 20, "dst_host_srv_count": 20,
        "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.05, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    },

    "Normal SSH Session": {
        "duration": 30, "protocol_type": "tcp", "service": "ssh",
        "flag": "SF", "src_bytes": 3000, "dst_bytes": 5000,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 1,
        "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 1, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 2, "srv_count": 2, "serror_rate": 0.0,
        "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 10, "dst_host_srv_count": 10,
        "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.1, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    },

    # ── DoS ATTACKS ───────────────────────────────────────────

    "DoS SYN Flood": {
        "duration": 0, "protocol_type": "tcp", "service": "http",
        "flag": "S0", "src_bytes": 0, "dst_bytes": 0,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
        "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 511, "srv_count": 511, "serror_rate": 1.0,
        "srv_serror_rate": 1.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 255, "dst_host_srv_count": 255,
        "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 1.0, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 1.0, "dst_host_srv_serror_rate": 1.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    },

    "DoS UDP Flood": {
        "duration": 0, "protocol_type": "udp", "service": "private",
        "flag": "SF", "src_bytes": 28, "dst_bytes": 0,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
        "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 511, "srv_count": 511, "serror_rate": 0.0,
        "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 255, "dst_host_srv_count": 255,
        "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 1.0, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    },

    "DoS ICMP Ping Flood": {
        "duration": 0, "protocol_type": "icmp", "service": "eco_i",
        "flag": "SF", "src_bytes": 1032, "dst_bytes": 0,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
        "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 511, "srv_count": 511, "serror_rate": 0.0,
        "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 255, "dst_host_srv_count": 255,
        "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 1.0, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    },

    "Land Attack": {
        "duration": 0, "protocol_type": "tcp", "service": "telnet",
        "flag": "S0", "src_bytes": 0, "dst_bytes": 0,
        "land": 1, "wrong_fragment": 0, "urgent": 0, "hot": 0,
        "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 1, "srv_count": 1, "serror_rate": 1.0,
        "srv_serror_rate": 1.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 255, "dst_host_srv_count": 255,
        "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 1.0, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 1.0, "dst_host_srv_serror_rate": 1.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    },

    # ── PROBE ATTACKS ─────────────────────────────────────────

    "Port Scan (Probe)": {
        "duration": 0, "protocol_type": "tcp", "service": "private",
        "flag": "REJ", "src_bytes": 0, "dst_bytes": 0,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
        "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 255, "srv_count": 1, "serror_rate": 0.0,
        "srv_serror_rate": 0.0, "rerror_rate": 1.0, "srv_rerror_rate": 1.0,
        "same_srv_rate": 0.0, "diff_srv_rate": 1.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 255, "dst_host_srv_count": 1,
        "dst_host_same_srv_rate": 0.0, "dst_host_diff_srv_rate": 1.0,
        "dst_host_same_src_port_rate": 0.0, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 1.0, "dst_host_srv_rerror_rate": 1.0
    },

    "Stealth Nmap Scan": {
        "duration": 0, "protocol_type": "tcp", "service": "private",
        "flag": "S0", "src_bytes": 0, "dst_bytes": 0,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
        "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 200, "srv_count": 2, "serror_rate": 1.0,
        "srv_serror_rate": 1.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 0.01, "diff_srv_rate": 0.99, "srv_diff_host_rate": 0.5,
        "dst_host_count": 255, "dst_host_srv_count": 2,
        "dst_host_same_srv_rate": 0.01, "dst_host_diff_srv_rate": 0.99,
        "dst_host_same_src_port_rate": 0.0, "dst_host_srv_diff_host_rate": 0.5,
        "dst_host_serror_rate": 1.0, "dst_host_srv_serror_rate": 1.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    },

    # ── R2L ATTACKS ───────────────────────────────────────────

    "R2L Brute Force (Telnet)": {
        "duration": 2, "protocol_type": "tcp", "service": "telnet",
        "flag": "SF", "src_bytes": 1200, "dst_bytes": 800,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0,
        "num_failed_logins": 5, "logged_in": 0, "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 1, "srv_count": 1, "serror_rate": 0.0,
        "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 10, "dst_host_srv_count": 10,
        "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.1, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    },

    "R2L FTP Write Attack": {
        "duration": 1, "protocol_type": "tcp", "service": "ftp_data",
        "flag": "SF", "src_bytes": 5000, "dst_bytes": 100,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 3,
        "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
        "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 3, "num_shells": 0, "num_access_files": 2,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 1,
        "count": 1, "srv_count": 1, "serror_rate": 0.0,
        "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 5, "dst_host_srv_count": 5,
        "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.2, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    },

    # ── U2R ATTACKS ───────────────────────────────────────────

    "U2R Root Privilege Escalation": {
        "duration": 0, "protocol_type": "tcp", "service": "telnet",
        "flag": "SF", "src_bytes": 1408, "dst_bytes": 1593,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 8,
        "num_failed_logins": 0, "logged_in": 1, "num_compromised": 1,
        "root_shell": 1, "su_attempted": 1, "num_root": 4,
        "num_file_creations": 0, "num_shells": 1, "num_access_files": 1,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 1, "srv_count": 1, "serror_rate": 0.0,
        "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 1, "dst_host_srv_count": 1,
        "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 1.0, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    },

    "U2R Buffer Overflow": {
        "duration": 0, "protocol_type": "tcp", "service": "telnet",
        "flag": "SF", "src_bytes": 480, "dst_bytes": 800,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 2,
        "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
        "root_shell": 1, "su_attempted": 0, "num_root": 1,
        "num_file_creations": 0, "num_shells": 1, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 1, "srv_count": 1, "serror_rate": 0.0,
        "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 2, "dst_host_srv_count": 2,
        "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.5, "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0, "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0
    }
}

# ── Run all tests ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

categories = {
    "NORMAL TRAFFIC" : ["Normal HTTP Traffic", "Normal FTP Session",
                        "Normal DNS (UDP)", "Normal SMTP (Email)", "Normal SSH Session"],
    "DoS ATTACKS"    : ["DoS SYN Flood", "DoS UDP Flood",
                        "DoS ICMP Ping Flood", "Land Attack"],
    "PROBE ATTACKS"  : ["Port Scan (Probe)", "Stealth Nmap Scan"],
    "R2L ATTACKS"    : ["R2L Brute Force (Telnet)", "R2L FTP Write Attack"],
    "U2R ATTACKS"    : ["U2R Root Privilege Escalation", "U2R Buffer Overflow"]
}

print("\n" + "="*65)
print(f"  {'NIDS — FULL TEST SUITE':^61}")
print("="*65)

total = correct = 0

for category, names in categories.items():
    print(f"\n  📂 {category}")
    print(f"  {'-'*61}")
    for name in names:
        sample = test_samples[name]
        result = predict_traffic(sample)
        label  = result["prediction"]
        conf   = result["confidence"]
        icon   = "🔴" if label == "Attack" else "🟢"

        # Expected: Normal for NORMAL TRAFFIC category, Attack for rest
        expected = "Normal" if category == "NORMAL TRAFFIC" else "Attack"
        correct += 1 if label == expected else 0
        total   += 1

        print(f"  {icon} {name:<38} {label:<8}  {conf:>6}%")

print("\n" + "="*65)
print(f"  ✅ Test Accuracy: {correct}/{total} ({(correct/total)*100:.1f}%)")
print("="*65 + "\n")

