"""
utils.py
========
Database helpers and metrics computation for the Multiclass NIDS.

DB schema supports:
  - Multiclass label (string like 'DDoS')
  - Prediction confidence
  - Attack score
  - Per-class probability JSON blob
"""

import json
import os
import sqlite3
import time
import traceback
from collections import Counter, defaultdict
from contextlib import closing

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
)

DB_PATH = os.path.join(os.path.dirname(__file__), 'nids.db')

CLASS_NAMES = ['Bot', 'Bruteforce', 'DDoS', 'DoS',
               'Infiltration', 'Normal', 'PortScan', 'WebAttack']
NORMAL_CLASS = 'Normal'


# ---------------------------------------------------------------------------
# DB Init
# ---------------------------------------------------------------------------
def init_db():
    """Create all required tables if they don't exist."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        c = conn.cursor()

        # Predictions table — multiclass aware
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              REAL,
                src_ip          TEXT,
                dst_ip          TEXT,
                features        TEXT,
                label_str       TEXT,
                label_idx       INTEGER,
                true_label_str  TEXT,
                true_label_idx  INTEGER,
                attack_score    REAL,
                confidence      REAL,
                class_proba     TEXT,
                latency_ms      REAL
            )
        ''')

        # Users table
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT
            )
        ''')

        conn.commit()
    print(f"[DB] ✅ Database initialised: {DB_PATH}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_prediction(record: dict):
    """Insert a prediction record into the DB."""
    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO predictions (
                    ts, src_ip, dst_ip, features,
                    label_str, label_idx,
                    true_label_str, true_label_idx,
                    attack_score, confidence, class_proba, latency_ms
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                record.get('ts', time.time()),
                record.get('src_ip'),
                record.get('dst_ip'),
                record.get('features'),
                record.get('label_str', 'Normal'),
                record.get('label_idx', 5),
                record.get('true_label_str', 'Normal'),
                record.get('true_label_idx', 5),
                record.get('attack_score', 0.0),
                record.get('confidence', 0.0),
                json.dumps(record.get('class_proba', {})),
                record.get('latency_ms', 0.0),
            ))
            conn.commit()
    except Exception as e:
        print(f"[DB LOG ERROR] {e}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _safe_float(v, default=0.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def get_metrics_data(session_start_ts: float = 0.0) -> dict:
    """
    Query DB and compute all metrics for the Analysis page.

    session_start_ts: only rows logged at/after this timestamp are counted.
    This must match the same boundary the Prediction page's live counters
    use, so both pages always report the same flow count.

    Returns JSON-serialisable dict.
    """
    empty = {
        'status': 'no_data',
        'count': 0,
        'basic_metrics': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0},
        'confusion_matrix': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0},
        'detection_rate': 0.0,
        'fpr': 0.0,
        'avg_latency': 0.0,
        'throughput_est': 0.0,
        'roc_data': {'fpr': [], 'tpr': []},
        'pr_data': {'precision': [], 'recall': []},
        'top_ips': [],
        'top_features': [],
        'class_distribution': {},
    }

    try:
        with closing(sqlite3.connect(DB_PATH)) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()

            c.execute('''
                SELECT ts, src_ip, features,
                       label_str, label_idx,
                       true_label_str, true_label_idx,
                       attack_score, confidence, latency_ms
                FROM predictions
                WHERE ts >= ?
                ORDER BY ts ASC
            ''', (session_start_ts,))
            rows = c.fetchall()

        count = len(rows)
        if count == 0:
            return empty

        # Build arrays
        y_pred_str, y_true_str = [], []
        y_pred_bin, y_true_bin = [], []
        y_score = []
        latencies = []
        timestamps = []
        src_ips = []
        features_list = []

        for r in rows:
            pred_str = r['label_str'] or 'Normal'
            true_str = r['true_label_str'] or 'Normal'
            y_pred_str.append(pred_str)
            y_true_str.append(true_str)

            y_pred_bin.append(0 if pred_str == NORMAL_CLASS else 1)
            y_true_bin.append(0 if true_str == NORMAL_CLASS else 1)

            y_score.append(_safe_float(r['attack_score']))
            latencies.append(_safe_float(r['latency_ms']))
            timestamps.append(_safe_float(r['ts']))
            src_ips.append(r['src_ip'])

            if r['features']:
                try:
                    parsed = json.loads(r['features'])
                    features_list.append(parsed if isinstance(parsed, dict) else {})
                except Exception:
                    features_list.append({})
            else:
                features_list.append({})

        # Binary metrics (binary=attack/benign)
        try:
            acc = float(accuracy_score(y_true_bin, y_pred_bin))
            prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
            rec = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
            f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))
        except Exception:
            acc = prec = rec = f1 = 0.0

        # Confusion matrix (binary)
        try:
            tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin).ravel()
        except Exception:
            tn = fp = fn = tp = 0
        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # ROC
        try:
            fpr_pts, tpr_pts, _ = roc_curve(y_true_bin, y_score)
            fpr_pts = [round(float(x), 6) for x in fpr_pts if np.isfinite(x)]
            tpr_pts = [round(float(x), 6) for x in tpr_pts if np.isfinite(x)]
        except Exception:
            fpr_pts = tpr_pts = []

        # PR curve
        try:
            prec_pts, rec_pts, _ = precision_recall_curve(y_true_bin, y_score)
            prec_pts = [round(float(x), 6) for x in prec_pts if np.isfinite(x)]
            rec_pts = [round(float(x), 6) for x in rec_pts if np.isfinite(x)]
        except Exception:
            prec_pts = rec_pts = []

        # Top attacking IPs
        top_ips_ctr = Counter(
            ip for ip, lbl in zip(src_ips, y_pred_str)
            if lbl != NORMAL_CLASS and ip
        )
        top_ips = [{'ip': ip, 'count': int(cnt)}
                   for ip, cnt in top_ips_ctr.most_common(5)]

        # Throughput
        avg_latency = float(np.mean(latencies)) if latencies else 0.0
        try:
            time_range = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
        except Exception:
            time_range = 0.0
        throughput = float(count / time_range) if time_range > 0 else 0.0

        # Multiclass distribution
        class_dist = Counter(y_pred_str)
        class_distribution = {cls: int(class_dist.get(cls, 0)) for cls in CLASS_NAMES}

        # Top features (mean diff attack vs benign)
        feat_attack_sum = defaultdict(float)
        feat_attack_cnt = defaultdict(int)
        feat_benign_sum = defaultdict(float)
        feat_benign_cnt = defaultdict(int)

        for feats, lbl in zip(features_list, y_pred_bin):
            if not isinstance(feats, dict):
                continue
            for k, v in feats.items():
                try:
                    fv = float(v)
                    if not np.isfinite(fv):
                        continue
                    if lbl == 1:
                        feat_attack_sum[k] += fv
                        feat_attack_cnt[k] += 1
                    else:
                        feat_benign_sum[k] += fv
                        feat_benign_cnt[k] += 1
                except Exception:
                    pass

        feat_scores = []
        for k in sorted(set(feat_attack_sum) | set(feat_benign_sum)):
            a_cnt = feat_attack_cnt.get(k, 0)
            b_cnt = feat_benign_cnt.get(k, 0)
            mean_a = feat_attack_sum[k] / a_cnt if a_cnt > 0 else 0.0
            mean_b = feat_benign_sum[k] / b_cnt if b_cnt > 0 else 0.0
            diff = abs(mean_a - mean_b)
            feat_scores.append({
                'feature': k,
                'score': round(diff, 4),
                'mean_attack': round(mean_a, 4),
                'mean_benign': round(mean_b, 4),
            })
        feat_scores.sort(key=lambda x: x['score'], reverse=True)
        top_features = feat_scores[:10]

        return {
            'status': 'ok',
            'count': count,
            'basic_metrics': {
                'accuracy':  round(acc, 4),
                'precision': round(prec, 4),
                'recall':    round(rec, 4),
                'f1':        round(f1, 4),
            },
            'confusion_matrix': {
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            },
            'detection_rate':  round(rec, 4),
            'fpr':             round(fpr_val, 4),
            'avg_latency':     round(avg_latency, 4),
            'throughput_est':  round(throughput, 4),
            'roc_data':        {'fpr': fpr_pts, 'tpr': tpr_pts},
            'pr_data':         {'precision': prec_pts, 'recall': rec_pts},
            'top_ips':         top_ips,
            'top_features':    top_features,
            'class_distribution': class_distribution,
        }

    except Exception as e:
        print(f"[METRICS ERROR] {e}")
        traceback.print_exc()
        err = dict(empty)
        err['status'] = 'error'
        err['message'] = str(e)
        return err
