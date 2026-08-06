"""
inference.py  —  Multiclass NIDS inference pipeline with SHAP explainability.

Architecture (unchanged):
  35 Features → RF → ISO → Meta Fusion → Meta Scaler → Meta Model → LabelEncoder

SHAP:
  TreeExplainer on the Random Forest model.
  Returns top-N feature contributions for every prediction.
  Gracefully disabled if shap is not installed.
"""

import os, time, traceback, warnings
import joblib, numpy as np, pandas as pd

import model_compat  # noqa – patches sklearn dtype before any model load

warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

CLASS_NAMES   = ['Bot','Bruteforce','DDoS','DoS','Infiltration','Normal','PortScan','WebAttack']
NORMAL_IDX    = 5   # index of 'Normal' class

THREAT_COLOURS = {
    'Normal':      '#22c55e',
    'Bot':         '#ef4444',
    'Bruteforce':  '#f97316',
    'DDoS':        '#ef4444',
    'DoS':         '#ef4444',
    'Infiltration':'#a855f7',
    'PortScan':    '#f59e0b',
    'WebAttack':   '#f97316',
}

# ── Try importing shap ────────────────────────────────────────────────────────
_SHAP_AVAILABLE = False
_SHAP_ERROR     = None
_shap           = None
try:
    import shap as _shap
    _SHAP_AVAILABLE = True
    print("[INFERENCE] ✅ shap library found — SHAP explainability enabled")
except ImportError as e:
    _SHAP_ERROR = f"shap not installed in this environment: {e}"
    print(f"[INFERENCE] ⚠️  {_SHAP_ERROR}")
    print("             Run: pip install shap  (make sure to use the venv pip)")
except Exception as e:
    _SHAP_ERROR = f"shap import failed ({type(e).__name__}): {e}"
    print(f"[INFERENCE] ⚠️  {_SHAP_ERROR}")


class EnsembleDetector:
    def __init__(self):
        self.rf = self.iso = self.meta_scaler = None
        self.meta_model = self.label_encoder = self.feature_names = None
        self.shap_explainer = None
        self._load()

    # ── Load ─────────────────────────────────────────────────────────────────
    def _load(self):
        print("[INFERENCE] Loading model artefacts …")
        self.rf           = model_compat.patch_rf(
                                joblib.load(os.path.join(MODELS_DIR,'rf_model_35feat.pkl')))
        print(f"[INFERENCE] ✅ RF  n_features={self.rf.n_features_in_} classes={self.rf.classes_}")

        self.iso          = model_compat.patch_iso(
                                joblib.load(os.path.join(MODELS_DIR,'iso_model_35feat.pkl')))
        print(f"[INFERENCE] ✅ ISO n_features={self.iso.n_features_in_}")

        self.meta_scaler  = joblib.load(os.path.join(MODELS_DIR,'meta_scaler_35feat.pkl'))
        print(f"[INFERENCE] ✅ Meta scaler n_features={self.meta_scaler.n_features_in_}")

        self.meta_model   = joblib.load(os.path.join(MODELS_DIR,'meta_model_35feat.pkl'))
        print(f"[INFERENCE] ✅ Meta model  classes={self.meta_model.classes_}")

        self.label_encoder= joblib.load(os.path.join(MODELS_DIR,'label_encoder.joblib'))
        print(f"[INFERENCE] ✅ Label encoder classes={self.label_encoder.classes_}")

        self.feature_names= joblib.load(os.path.join(MODELS_DIR,'selected_features_35feat.pkl'))
        print(f"[INFERENCE] ✅ Features count={len(self.feature_names)}")

        assert len(self.feature_names) == 35
        assert self.rf.n_features_in_  == 35
        assert self.iso.n_features_in_ == 35
        assert self.meta_scaler.n_features_in_ == 10
        print("[INFERENCE] ✅ Sanity checks passed.")

        # ── SHAP explainer (TreeExplainer on RF) ─────────────────────────────
        global _SHAP_ERROR
        if _SHAP_AVAILABLE:
            try:
                # tree_path_dependent avoids needing background data and is
                # faster/more compatible with older sklearn-trained RF models
                self.shap_explainer = _shap.TreeExplainer(
                    self.rf,
                    feature_perturbation="tree_path_dependent"
                )
                print("[INFERENCE] ✅ SHAP TreeExplainer initialised on RF model.")
            except Exception as e:
                _SHAP_ERROR = f"TreeExplainer init failed ({type(e).__name__}): {e}"
                print(f"[INFERENCE] ⚠️  {_SHAP_ERROR}")
                self.shap_explainer = None
        else:
            self.shap_explainer = None

    # ── Feature vector ────────────────────────────────────────────────────────
    def _build_X(self, flow_row: dict) -> pd.DataFrame:
        vec = {}
        for feat in self.feature_names:
            try:
                vec[feat] = float(flow_row.get(feat, 0) or 0)
            except (TypeError, ValueError):
                vec[feat] = 0.0
        return pd.DataFrame([vec])[self.feature_names]

    # ── SHAP values for one sample ────────────────────────────────────────────
    def _compute_shap(self, X: pd.DataFrame, pred_class_idx: int, top_n: int = 8):
        """
        Return top_n SHAP feature contributions for the predicted class.
        Returns list of {feature, value, shap_value, direction} dicts.
        direction: 'up' (pushes toward attack) or 'down' (pushes toward normal).
        """
        if self.shap_explainer is None:
            return []
        try:
            # shap_values: for RF multiclass returns list[n_classes] of (1,35) arrays
            shap_vals = self.shap_explainer.shap_values(X)

            if isinstance(shap_vals, list):
                # Multi-output: pick the predicted class
                sv = np.array(shap_vals[pred_class_idx])[0]   # shape (35,)
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                # Some versions return (1, 35, n_classes)
                sv = shap_vals[0, :, pred_class_idx]
            else:
                sv = np.array(shap_vals)[0]

            feature_vals = X.values[0]   # shape (35,)
            names        = self.feature_names

            # Sort by |shap_value| descending
            order = np.argsort(np.abs(sv))[::-1][:top_n]

            result = []
            for i in order:
                result.append({
                    'feature':    names[i],
                    'value':      round(float(feature_vals[i]), 4),
                    'shap_value': round(float(sv[i]), 6),
                    'direction':  'up' if sv[i] > 0 else 'down',
                })
            return result
        except Exception as e:
            import traceback as _tb
            print(f"[SHAP] compute_shap error: {type(e).__name__}: {e}")
            _tb.print_exc()
            return []

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, flow_row: dict, compute_shap: bool = False) -> dict:
        t0 = time.time()
        try:
            X = self._build_X(flow_row)

            # 1. RF → 8-class proba
            rf_proba = self.rf.predict_proba(X)          # (1,8)

            # 2. ISO → score + decision
            iso_score  = self.iso.score_samples(X)       # (1,)
            iso_dec    = self.iso.decision_function(X)   # (1,)
            iso_norm   = float(np.interp(iso_score[0], [-0.5, 0.0], [1.0, 0.0]))

            # 3. Meta fusion: rf_proba(8) + iso_norm(1) + iso_dec(1) = 10
            meta_X      = np.hstack([rf_proba, [[iso_norm]], iso_dec.reshape(1,-1)])
            meta_scaled = self.meta_scaler.transform(meta_X)

            # 4. Meta model prediction
            pred_idx   = int(self.meta_model.predict(meta_scaled)[0])
            meta_proba = self.meta_model.predict_proba(meta_scaled)[0]

            # 5. Label
            label_str  = str(self.label_encoder.inverse_transform([pred_idx])[0])
            is_attack  = (pred_idx != NORMAL_IDX)
            confidence = float(meta_proba[pred_idx])
            attack_score = round(1.0 - float(meta_proba[NORMAL_IDX]), 4)

            class_proba = {CLASS_NAMES[i]: round(float(meta_proba[i]),4)
                           for i in range(len(CLASS_NAMES))}

            # 6. SHAP — only when explicitly requested (expensive: ~10s per flow on RF-300)
            shap_top = self._compute_shap(X, pred_idx, top_n=8) if compute_shap else []

            return {
                'label_str':    label_str,
                'label_idx':    pred_idx,
                'label':        1 if is_attack else 0,
                'is_attack':    is_attack,
                'confidence':   round(confidence, 4),
                'attack_score': attack_score,
                'class_proba':  class_proba,
                'colour':       THREAT_COLOURS.get(label_str, '#ffffff'),
                'latency_ms':   round((time.time()-t0)*1000, 2),
                'shap_values':  shap_top,
                'shap_available': len(shap_top) > 0 and compute_shap,
            }

        except Exception as exc:
            print(f"[INFERENCE ERROR] {exc}")
            traceback.print_exc()
            return {
                'error':        str(exc),
                'label':        0, 'label_str':'Normal', 'label_idx': NORMAL_IDX,
                'is_attack':    False, 'attack_score':0.0, 'confidence':0.0,
                'class_proba':  {}, 'colour':'#22c55e',
                'latency_ms':   round((time.time()-t0)*1000,2),
                'shap_values':  [], 'shap_available': False,
            }


# ── Singleton ─────────────────────────────────────────────────────────────────
print("[INFERENCE] Initialising detector …")
_detector = EnsembleDetector()
print("[INFERENCE] ✅ Detector ready.")


def predict_flow(flow_row: dict) -> dict:
    """Used by replay and live flow processing. SHAP is NOT computed here for speed."""
    return _detector.predict(flow_row, compute_shap=False)

def get_feature_names() -> list:
    return list(_detector.feature_names)

def get_class_names() -> list:
    return CLASS_NAMES


def get_shap_for_flow(flow_row: dict, top_n: int = 35) -> dict:
    """Compute SHAP values for all 35 features. Used by the SHAP explainer page."""
    # Surface the real error if shap failed to load
    if not _SHAP_AVAILABLE or _detector.shap_explainer is None:
        reason = _SHAP_ERROR or "SHAP explainer not initialised"
        return {
            'available': False,
            'error': reason,
            'shap_installed': _SHAP_AVAILABLE,
        }
    try:
        X   = _detector._build_X(flow_row)
        res = _detector.predict(flow_row, compute_shap=True)
        pred_idx  = res.get('label_idx', 5)
        all_shap  = _detector._compute_shap(X, pred_idx, top_n=35)
        if not all_shap:
            return {
                'available': False,
                'error': 'SHAP computation returned empty — check server log for details',
                'shap_installed': True,
            }
        return {
            'available':     True,
            'all_shap':      all_shap,
            'label_str':     res.get('label_str', 'Normal'),
            'label_idx':     pred_idx,
            'is_attack':     res.get('is_attack', False),
            'class_proba':   res.get('class_proba', {}),
            'confidence':    res.get('confidence', 0.0),
            'attack_score':  res.get('attack_score', 0.0),
            'shap_installed': True,
        }
    except Exception as e:
        import traceback as _tb
        _tb.print_exc()
        return {'available': False, 'error': f'{type(e).__name__}: {e}', 'shap_installed': _SHAP_AVAILABLE}
