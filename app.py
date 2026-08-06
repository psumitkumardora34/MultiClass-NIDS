"""
app.py
======
Flask + Socket.IO server for the Multiclass Network Intrusion Detection System.

Key fixes vs original:
  - REPLAY uses data/replay_demo.parquet (not hardcoded Windows path)
  - Replay reads the correct 35 features expected by the model
  - Flow handler sends multiclass result (label_str, class_proba, colour)
  - Old binary-only 'true_label = 0/1' logic replaced with multiclass
  - Dead REPLAY_CSV_PATH variable removed
  - 'alarm' event now carries the attack class name
  - set_params socket event removed (no longer needed — model has no alpha/beta)
  - All socket events guarded against unauthenticated calls
  - GeoIP handled gracefully
"""

import json
import os
import time
import sqlite3
import traceback
import numpy as np
import pandas as pd
import geoip2.database

from flask import (
    Flask, render_template, request, redirect,
    url_for, jsonify, session, g
)
from flask_socketio import SocketIO, emit
from werkzeug.security import generate_password_hash, check_password_hash
from threading import Lock
from functools import wraps

from inference import predict_flow, get_feature_names, get_class_names, get_shap_for_flow
from utils import init_db, log_prediction, DB_PATH, get_metrics_data

# ============================================================
# App setup
# ============================================================
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET', 'nids-secret-2024-change-me')

socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')
init_db()

# ============================================================
# GeoIP
# ============================================================
_GEOIP_DB = os.path.join(os.path.dirname(__file__), 'GeoLite2-City.mmdb')
try:
    geoip_reader = geoip2.database.Reader(_GEOIP_DB)
    print(f"[GEO] ✅ GeoIP DB loaded")
except FileNotFoundError:
    print("[GEO] ⚠️  GeoLite2-City.mmdb not found — using random coordinates")
    geoip_reader = None


def _get_geo(ip: str) -> dict:
    if geoip_reader:
        try:
            resp = geoip_reader.city(ip)
            return {'lat': resp.location.latitude, 'lon': resp.location.longitude}
        except Exception:
            pass
    return {
        'lat': float(np.random.uniform(-60, 70)),
        'lon': float(np.random.uniform(-180, 180))
    }


# ============================================================
# Session boundary (persisted so it survives server restarts)
# ============================================================
# Both the Prediction page (_session_stats, in RAM) and the Analysis page
# (DB query) must always agree on the same flow count. RAM resets on every
# server restart but the DB does not — so instead of trusting RAM as the
# source of truth, we treat the DATABASE (filtered to rows at/after
# _session_start_ts) as the single source of truth, and rehydrate RAM from
# it on startup. _session_start_ts only moves forward on an explicit
# "Reset View" click — never on a normal replay start/stop.
_SESSION_META_PATH = os.path.join(os.path.dirname(__file__), 'data', 'session_meta.json')

def _load_session_start_ts() -> float:
    try:
        with open(_SESSION_META_PATH, 'r') as f:
            return float(json.load(f).get('session_start_ts', 0.0))
    except Exception:
        return 0.0

def _save_session_start_ts(ts: float):
    try:
        os.makedirs(os.path.dirname(_SESSION_META_PATH), exist_ok=True)
        with open(_SESSION_META_PATH, 'w') as f:
            json.dump({'session_start_ts': ts}, f)
    except Exception as e:
        print(f"[SESSION META ERROR] {e}")

_session_start_ts = _load_session_start_ts()


# ============================================================
# Replay state
# ============================================================
_replay_thread = None
_thread_lock = Lock()
_replay_running = False

# Live session counters — persist across page navigations.
# Rehydrated from the DB at startup (see _rehydrate_session_stats below)
# so a server restart can never make this disagree with the Analysis page.
_session_stats = {
    'total': 0, 'attacks': 0, 'normal': 0,
    'class_counts': {c: 0 for c in
                     ['Bot','Bruteforce','DDoS','DoS','Infiltration','Normal','PortScan','WebAttack']},
    'replay_running': False,
}

def _rehydrate_session_stats():
    """Recompute _session_stats from the DB (rows at/after _session_start_ts).
    Called at startup so RAM counters always match what the Analysis page
    will show, even after a server restart."""
    global _session_stats
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT label_str FROM predictions WHERE ts >= ?", (_session_start_ts,)
        )
        rows = cur.fetchall()
        conn.close()

        total = len(rows)
        attacks = sum(1 for r in rows if r['label_str'] != 'Normal')
        normal = total - attacks
        class_counts = {c: 0 for c in _session_stats['class_counts']}
        for r in rows:
            if r['label_str'] in class_counts:
                class_counts[r['label_str']] += 1

        _session_stats['total'] = total
        _session_stats['attacks'] = attacks
        _session_stats['normal'] = normal
        _session_stats['class_counts'] = class_counts
        print(f"[SESSION] 🔄 Rehydrated from DB — {total} flows since session start")
    except Exception as e:
        print(f"[SESSION REHYDRATE ERROR] {e}")

_rehydrate_session_stats()

def reset_replay_session():
    """
    Start a completely fresh replay session (explicit user action only).

    Clears:
      - live counters
      - prediction history used by Analysis page
      - moves the session boundary forward so old rows (if any survive)
        never get counted again
    """
    global _session_stats, _session_start_ts

    _session_stats['total'] = 0
    _session_stats['attacks'] = 0
    _session_stats['normal'] = 0
    _session_stats['replay_running'] = False

    for k in _session_stats['class_counts']:
        _session_stats['class_counts'][k] = 0

    _session_start_ts = time.time()
    _save_session_start_ts(_session_start_ts)

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()

        print("[RESET] Prediction history cleared")
    except Exception as e:
        print(f"[RESET ERROR] {e}")
REPLAY_PARQUET       = os.path.join(os.path.dirname(__file__), 'data', 'replay_demo.parquet')
REPLAY_CSV_FALLBACK  = os.path.join(os.path.dirname(__file__), 'data', 'replay_fallback.csv')

# ============================================================
# DB helper
# ============================================================
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
    return db


@app.teardown_appcontext
def close_db(exc):
    db = getattr(g, '_database', None)
    if db:
        db.close()


# ============================================================
# Auth helpers
# ============================================================
def login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapped


# ============================================================
# Routes
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        conn = get_db()
        user = conn.execute(
            'SELECT * FROM users WHERE username=?', (username,)
        ).fetchone()
        if user and check_password_hash(user[2], password):
            session['user'] = user[1]
            return redirect(url_for('prediction'))
        error = 'Invalid username or password.'
    return render_template('login.html', error=error)


@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            error = 'Username and password are required.'
        else:
            hashed = generate_password_hash(password, method='pbkdf2:sha256')
            try:
                conn = get_db()
                conn.execute(
                    'INSERT INTO users (username, password) VALUES (?,?)',
                    (username, hashed)
                )
                conn.commit()
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                error = 'Username already exists.'
    return render_template('register.html', error=error)


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))


@app.route('/prediction')
@login_required
def prediction():
    return render_template('prediction.html', user=session.get('user'))


@app.route('/analysis')
@login_required
def analysis():
    return render_template('analysis.html', user=session.get('user'))


@app.route('/explainer')
@login_required
def explainer():
    return render_template('explainer.html', user=session.get('user'))


@app.route('/api/explain', methods=['POST'])
@login_required
def api_explain():
    """Run SHAP explanation on a submitted flow dict."""
    import json as _json
    try:
        data = request.get_json(force=True) or {}
        result = get_shap_for_flow(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'available': False, 'error': str(e)}), 500


@app.route('/api/last_flow_shap')
@login_required
def api_last_flow_shap():
    """Return SHAP values for the most recently processed flow (from DB)."""
    import sqlite3 as _sql, json as _json
    from contextlib import closing as _closing
    try:
        with _closing(_sql.connect(DB_PATH)) as conn:
            conn.row_factory = _sql.Row
            row = conn.execute(
                'SELECT id, features, label_str, label_idx, attack_score, confidence, ts '
                'FROM predictions ORDER BY ts DESC LIMIT 1'
            ).fetchone()
        if not row or not row['features']:
            return jsonify({'available': False, 'error': 'No predictions yet'})
        flow = _json.loads(row['features'])
        result = get_shap_for_flow(flow)
        result['db_label'] = row['label_str']
        result['db_score'] = row['attack_score']
        result['db_conf']  = row['confidence']
        result['flow_id']  = row['id']
        result['flow_ts']  = row['ts']
        return jsonify(result)
    except Exception as e:
        return jsonify({'available': False, 'error': str(e)}), 500


@app.route('/api/explain/<int:flow_id>')
@login_required
def api_explain_by_id(flow_id):
    """
    Return SHAP explanation for any historical flow by its database ID.
    This enables analysts to explain ANY previously predicted flow,
    not just the most recent one.
    """
    import sqlite3 as _sql, json as _json
    from contextlib import closing as _closing
    try:
        with _closing(_sql.connect(DB_PATH)) as conn:
            conn.row_factory = _sql.Row
            row = conn.execute(
                'SELECT id, ts, src_ip, dst_ip, features, label_str, label_idx, '
                '       true_label_str, attack_score, confidence, class_proba, latency_ms '
                'FROM predictions WHERE id = ?',
                (flow_id,)
            ).fetchone()
        if not row:
            return jsonify({'available': False, 'error': f'Flow #{flow_id} not found'}), 404
        if not row['features']:
            return jsonify({'available': False, 'error': f'Flow #{flow_id} has no stored features'}), 400

        flow = _json.loads(row['features'])
        result = get_shap_for_flow(flow)

        # Enrich with all stored metadata
        result['flow_id']       = row['id']
        result['flow_ts']       = row['ts']
        result['src_ip']        = row['src_ip']
        result['dst_ip']        = row['dst_ip']
        result['db_label']      = row['label_str']
        result['db_label_idx']  = row['label_idx']
        result['true_label']    = row['true_label_str']
        result['db_score']      = row['attack_score']
        result['db_conf']       = row['confidence']
        result['latency_ms']    = row['latency_ms']
        result['stored_proba']  = _json.loads(row['class_proba']) if row['class_proba'] else {}
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'available': False, 'error': str(e)}), 500


@app.route('/api/predictions')
@login_required
def api_predictions_history():
    """
    Return paginated prediction history for the Explainer page history table.
    Supports: page, per_page, class_filter, attacks_only, search (src_ip).
    """
    import sqlite3 as _sql
    from contextlib import closing as _closing

    page        = max(1, request.args.get('page',       1,    type=int))
    per_page    = min(50, request.args.get('per_page',  20,   type=int))
    cls_filter  = request.args.get('class_filter', '')
    attacks_only= request.args.get('attacks_only', 'false').lower() == 'true'
    search_ip   = request.args.get('search', '').strip()

    offset = (page - 1) * per_page

    # Build WHERE clause
    conditions = []
    params     = []
    if cls_filter:
        conditions.append('label_str = ?'); params.append(cls_filter)
    if attacks_only:
        conditions.append("label_str != 'Normal'")
    if search_ip:
        conditions.append('src_ip LIKE ?'); params.append(f'%{search_ip}%')

    where = ('WHERE ' + ' AND '.join(conditions)) if conditions else ''

    try:
        with _closing(_sql.connect(DB_PATH)) as conn:
            conn.row_factory = _sql.Row
            total = conn.execute(
                f'SELECT COUNT(*) FROM predictions {where}', params
            ).fetchone()[0]

            rows = conn.execute(
                f'''SELECT id, ts, src_ip, dst_ip, label_str, label_idx,
                           true_label_str, attack_score, confidence, latency_ms
                    FROM predictions {where}
                    ORDER BY ts DESC
                    LIMIT ? OFFSET ?''',
                params + [per_page, offset]
            ).fetchall()

        import time as _time
        def fmt_ts(ts):
            try:
                import datetime
                return datetime.datetime.fromtimestamp(float(ts)).strftime('%H:%M:%S')
            except Exception:
                return str(ts)

        flows = [{
            'id':          r['id'],
            'ts':          r['ts'],
            'ts_fmt':      fmt_ts(r['ts']),
            'src_ip':      r['src_ip'],
            'dst_ip':      r['dst_ip'],
            'label_str':   r['label_str'],
            'label_idx':   r['label_idx'],
            'true_label':  r['true_label_str'],
            'attack_score': round(float(r['attack_score'] or 0), 3),
            'confidence':  round(float(r['confidence']   or 0), 3),
            'latency_ms':  round(float(r['latency_ms']   or 0), 1),
            'is_attack':   r['label_str'] != 'Normal',
        } for r in rows]

        return jsonify({
            'flows':    flows,
            'total':    total,
            'page':     page,
            'per_page': per_page,
            'pages':    max(1, (total + per_page - 1) // per_page),
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================
# API
# ============================================================
@app.route('/api/metrics_data')
@login_required
def metrics_data():
    return jsonify(get_metrics_data(_session_start_ts))


@app.route('/api/classes')
def api_classes():
    return jsonify({'classes': get_class_names()})


@app.route('/api/session_stats')
@login_required
def api_session_stats():
    """Return live session counters so the page can restore state on reload."""
    return jsonify(_session_stats)


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'time': time.time()})


# ============================================================
# Replay thread
# ============================================================
def _replay_worker(speed: float):
    """Background task: read replay_demo.parquet and emit flows via Socket.IO."""
    global _replay_running

    print(f"[REPLAY] ▶️  Starting replay at {speed}x from: {REPLAY_PARQUET}")

    # ── Load data (parquet preferred, CSV fallback) ──────────────────────
    df = None
    if os.path.exists(REPLAY_PARQUET):
        try:
            df = pd.read_parquet(REPLAY_PARQUET)
            df.columns = df.columns.str.strip()
            print(f"[REPLAY] ✅ Loaded parquet: {len(df)} rows")
        except Exception as e:
            print(f"[REPLAY] ⚠️  Parquet load failed ({e}), trying CSV fallback …")
            df = None

    if df is None:
        if os.path.exists(REPLAY_CSV_FALLBACK):
            try:
                df = pd.read_csv(REPLAY_CSV_FALLBACK)
                df.columns = df.columns.str.strip()
                print(f"[REPLAY] ✅ Loaded CSV fallback: {len(df)} rows")
            except Exception as e:
                print(f"[REPLAY ERROR] ❌ CSV fallback failed: {e}")
                _replay_running = False
                socketio.emit('replay_error', {'message': f'Cannot load replay data: {e}'})
                return
        else:
            msg = ('No replay data found. Install pyarrow (pip install pyarrow) '
                   'or ensure data/replay_fallback.csv exists.')
            print(f"[REPLAY ERROR] ❌ {msg}")
            _replay_running = False
            socketio.emit('replay_error', {'message': msg})
            return

    # Get the 35 feature names the model expects
    model_features = get_feature_names()

    # Validate columns
    available = set(df.columns)
    missing_feats = [f for f in model_features if f not in available]
    if missing_feats:
        print(f"[REPLAY WARN] Features missing from parquet (will be 0): {missing_feats}")

    # Handle Label column
    label_col = None
    for candidate in ['Label', 'label', 'CLASS', 'Class']:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        df['Label'] = 'Normal'
        label_col = 'Label'

    records = df.to_dict(orient='records')
    total = len(records)
    print(f"[REPLAY] ✅ {total} flows ready. Label col: '{label_col}'")
    socketio.emit('replay_started', {'total': total})

    # Adaptive sleep: target_interval is desired gap between flows.
    # We subtract actual inference time so high speed settings are honoured.
    target_interval = 1.0 / max(float(speed), 0.1)
    processed = 0

    for row in records:
        if not _replay_running:
            print("[REPLAY] ⛔ Stopped by user.")
            break

        flow_start = time.time()

        try:
            # Build flow dict with the 35 model features
            flow_data = {}
            for feat in model_features:
                val = row.get(feat, 0)
                try:
                    flow_data[feat] = float(val) if val is not None else 0.0
                except (TypeError, ValueError):
                    flow_data[feat] = 0.0

            # True label from dataset
            true_label_raw = str(row.get(label_col, 'Normal')).strip()
            true_label_str = _normalise_label(true_label_raw)

            # Random IPs
            src_ip = _random_ip()
            dst_ip = _random_ip()

            # Run prediction — SHAP is NOT computed during replay (would add ~10s/flow)
            _process_flow(flow_data, src_ip, dst_ip, true_label_str)

            processed += 1
            if processed % 100 == 0:
                inf_ms = (time.time() - flow_start) * 1000
                print(f"[REPLAY] {processed}/{total}  inference={inf_ms:.0f}ms  speed={speed}x")

        except Exception as e:
            print(f"[REPLAY FLOW ERROR] {e}")
            traceback.print_exc()

        # Sleep only for the time remaining after inference
        elapsed = time.time() - flow_start
        socketio.sleep(max(0.0, target_interval - elapsed))

    print(f"[REPLAY] ✅ Complete — {processed}/{total} flows processed.")
    _replay_running = False
    socketio.emit('replay_done', {'processed': processed, 'total': total})


def _normalise_label(raw: str) -> str:
    """Map raw dataset labels to model class names."""
    raw_l = raw.lower().strip()
    mapping = {
        'benign': 'Normal',
        'normal': 'Normal',
        'ddos': 'DDoS',
        'dos': 'DoS',
        'dos hulk': 'DoS',
        'dos goldeneye': 'DoS',
        'dos slowloris': 'DoS',
        'dos slowhttptest': 'DoS',
        'heartbleed': 'DoS',
        'bot': 'Bot',
        'bruteforce': 'Bruteforce',
        'brute force': 'Bruteforce',
        'ftp-patator': 'Bruteforce',
        'ssh-patator': 'Bruteforce',
        'infiltration': 'Infiltration',
        'portscan': 'PortScan',
        'port scan': 'PortScan',
        'web attack – brute force': 'WebAttack',
        'web attack – sql injection': 'WebAttack',
        'web attack – xss': 'WebAttack',
        'webattack': 'WebAttack',
    }
    return mapping.get(raw_l, 'Normal')


def _random_ip() -> str:
    return (f"{np.random.randint(11,223)}."
            f"{np.random.randint(0,255)}."
            f"{np.random.randint(0,255)}."
            f"{np.random.randint(1,255)}")


# ============================================================
# Core flow processing (used by replay AND live new_flow event)
# ============================================================
def _process_flow(flow_data: dict, src_ip: str, dst_ip: str,
                  true_label_str: str = 'Normal'):
    """
    Run prediction on one flow, log to DB, emit Socket.IO events.
    """
    ts = time.time()

    try:
        res = predict_flow(flow_data)

        if 'error' in res and not res.get('label_str'):
            socketio.emit('error', {'message': res['error']})
            return

        label_str = res.get('label_str', 'Normal')
        label_idx = res.get('label_idx', 5)
        is_attack = res.get('is_attack', False)

        # Update live session counters
        _session_stats['total'] += 1
        if is_attack:
            _session_stats['attacks'] += 1
        else:
            _session_stats['normal'] += 1
        if label_str in _session_stats['class_counts']:
            _session_stats['class_counts'][label_str] += 1
        _session_stats['replay_running'] = _replay_running
        confidence = res.get('confidence', 0.0)
        attack_score = res.get('attack_score', 0.0)
        class_proba = res.get('class_proba', {})
        colour = res.get('colour', '#22c55e')
        latency_ms = res.get('latency_ms', 0.0)

        # Geo
        geo = _get_geo(src_ip)

        # Emit flow result
        out = {
            'ts':             ts,
            'src_ip':         src_ip,
            'dst_ip':         dst_ip,
            'label':          1 if is_attack else 0,
            'label_str':      label_str,
            'is_attack':      is_attack,
            'attack_score':   attack_score,
            'confidence':     confidence,
            'class_proba':    class_proba,
            'colour':         colour,
            'geo':            geo,
            'shap_values':    res.get('shap_values', []),
            'shap_available': res.get('shap_available', False),
        }
        socketio.emit('flow_result', out)

        # Alarm if attack
        if is_attack:
            socketio.emit('alarm', {**out, 'attack_type': label_str})

        # DB logging
        true_label_idx = 5  # default Normal
        CLASS_NAMES = get_class_names()
        if true_label_str in CLASS_NAMES:
            true_label_idx = CLASS_NAMES.index(true_label_str)

        try:
            log_prediction({
                'ts':             ts,
                'src_ip':         src_ip,
                'dst_ip':         dst_ip,
                'features':       json.dumps(flow_data),
                'label_str':      label_str,
                'label_idx':      label_idx,
                'true_label_str': true_label_str,
                'true_label_idx': true_label_idx,
                'attack_score':   attack_score,
                'confidence':     confidence,
                'class_proba':    class_proba,
                'latency_ms':     latency_ms,
            })
        except Exception as e:
            print(f"[DB LOG ERROR] {e}")

    except Exception as e:
        print(f"[FLOW ERROR] {e}")
        traceback.print_exc()
        socketio.emit('error', {'message': str(e)})


# ============================================================
# Socket.IO event handlers
# ============================================================
@socketio.on('connect')
def on_connect():
    print(f"[SOCKET] Client connected — session user: {session.get('user', 'anonymous')}")


@socketio.on('disconnect')
def on_disconnect():
    print("[SOCKET] Client disconnected")


@socketio.on('start_replay')
def start_replay(data):

    if 'user' not in session:
        emit('error', {'message': 'Not authenticated'})
        return

    global _replay_thread, _replay_running

    with _thread_lock:
        speed = float(data.get('speed', 5.0))
        speed = max(0.1, min(speed, 100.0))  # clamp

        if _replay_running:
            print("[SOCKET] ⚠️  Replay already running — stopping first")
            _replay_running = False
            socketio.sleep(0.3)

        _replay_running = True
        # NOTE: counters/DB are intentionally NOT reset here anymore — flows
        # from this replay accumulate on top of any previous session totals.
        # Totals only reset when the user explicitly clicks "Reset View".
        _session_stats['replay_running'] = True
        _replay_thread = socketio.start_background_task(_replay_worker, speed)
        print(f"[SOCKET] 🚀 Replay thread started at {speed}x speed")


@socketio.on('reset_session')
def reset_session():
    """Explicit reset, triggered only by the 'Reset View' button."""
    reset_replay_session()
    emit('session_reset', {'ok': True})
    print("[SOCKET] 🗑 Session explicitly reset by user")


@socketio.on('stop_replay')
def stop_replay():
    if 'user' not in session:
        return
    global _replay_running
    _replay_running = False
    _session_stats['replay_running'] = False
    print("[SOCKET] ⛔ Replay stopped by user")
    emit('replay_stopped', {})


@socketio.on('new_flow')
def handle_new_flow(data):
    """Accept a live flow dict from an external client or UI."""
    if 'user' not in session:
        emit('error', {'message': 'Not authenticated'})
        return

    src_ip = data.pop('src_ip', _random_ip())
    dst_ip = data.pop('dst_ip', _random_ip())
    true_label = data.pop('true_label', 'Normal')

    _process_flow(data, src_ip, dst_ip, true_label)


# ============================================================
# Entry point
# ============================================================
if __name__ == '__main__':
    print("🚀 NIDS Multiclass IDS — Flask-SocketIO server")
    print("   http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
