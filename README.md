# 🛡️ ML-Driven Network Intrusion Detection System (NIDS)

A machine learning-based NIDS trained on the **NSL-KDD dataset** using **XGBoost**,
achieving **99.65% accuracy** and **99.64% F1-score**. Includes a full Streamlit dashboard for real-time traffic analysis.

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 99.65% |
| F1-Score | 99.64% |
| Model | XGBoost |
| Dataset | NSL-KDD |

---

## 📁 Project Structure
```
NIDS/
├── nids_model/
│   ├── metadata.json        ← Model metadata
│   └── feature_names.json   ← Feature order
├── app.py                   ← Streamlit dashboard
├── predict.py               ← CLI inference script
├── requirements.txt         ← Dependencies
└── README.md
```

> ⚠️ The `.pkl` model files are too large for GitHub. Download from Google Drive below.

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Sharanya1804/NIDS.git
cd NIDS
```

### 2. Create virtual environment
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1      # Windows
source venv/bin/activate          # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download model files
Download `nids_model.zip` from **[Google Drive](#)** ← replace with your link  
Extract it into the project root so you have a `nids_model/` folder.

---

## 🚀 Usage

### Run Streamlit Dashboard
```bash
streamlit run app.py
```

### Run CLI Batch Test
```bash
python predict.py
```

---

## 🔍 Attack Categories Detected

| Category | Description | Examples |
|----------|-------------|---------|
| DoS | Denial of Service | SYN Flood, ICMP Flood |
| Probe | Network Scanning | Port Scan, Nmap |
| R2L | Remote to Local | Brute Force, FTP Write |
| U2R | User to Root | Buffer Overflow, Rootkit |
| Normal | Benign Traffic | HTTP, FTP, DNS, SMTP |

---

## 🛠️ Tech Stack

- **Python 3.11**
- **XGBoost** — primary classifier
- **Scikit-learn** — preprocessing & evaluation
- **Streamlit** — web dashboard
- **Plotly** — interactive charts
- **Pandas / NumPy** — data processing

---
