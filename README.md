# Multiclass Network Intrusion Detection System (NIDS)

## Architecture
```
35 Features → RF (35feat) → ISO (35feat) → Meta Fusion → Meta Scaler → Meta Model → Label Encoder → 8-class Prediction
```

### Classes
| Index | Class       | Type   |
|-------|-------------|--------|
| 0     | Bot         | Attack |
| 1     | Bruteforce  | Attack |
| 2     | DDoS        | Attack |
| 3     | DoS         | Attack |
| 4     | Infiltration| Attack |
| 5     | Normal      | Benign |
| 6     | PortScan    | Attack |
| 7     | WebAttack   | Attack |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
python app.py
```

Open: http://localhost:5000

## Usage
1. Register an account at `/register`
2. Login at `/login`
3. Go to **Prediction** page
4. Click **▶ Start Replay** to begin
5. Watch the dashboard update in real-time
6. After replay, visit **Analysis** page for metrics

## Model Files (do not modify)
```
models/
  rf_model_35feat.pkl        — RandomForestClassifier (35 features, 8 classes)
  iso_model_35feat.pkl       — IsolationForest (35 features)
  meta_model_35feat.pkl      — LogisticRegression meta-classifier (10 inputs, 8 outputs)
  meta_scaler_35feat.pkl     — StandardScaler for meta inputs
  label_encoder.joblib       — LabelEncoder for 8 class names
  selected_features_35feat.pkl — List of 35 feature names in training order

data/
  replay_demo.parquet        — CICIDS2017 replay dataset
```

## Key Files Modified
- `model_compat.py` — sklearn 1.2.2 → 1.8.0 compatibility patches
- `inference.py`    — Correct multiclass pipeline implementation  
- `utils.py`        — Multiclass-aware DB schema and metrics
- `app.py`          — Fixed replay (parquet), multiclass flow handler, Socket.IO
- `templates/prediction.html` — Multiclass live dashboard
- `templates/analysis.html`   — Multiclass analysis with class distribution

## Troubleshooting
- If models fail to load: ensure scikit-learn >= 1.4.0 is installed
- If parquet fails: ensure pyarrow is installed
- If socket events don't arrive: check browser console for errors
