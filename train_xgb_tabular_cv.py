# train_xgb_tabular_cv.py
import json, numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import joblib
from utils_sampling import make_one_to_one_sample  # same helper used in LGBM script

# ---------- Inputs ----------
X_TAB = Path("X_tab.npy")
FN_TAB= Path("feature_names_tab.json")

# Optional blocks
X_SCO = Path("X_scores.npy");      FN_SCO= Path("feature_names_scores.json")
X_PATH= Path("X_pathway.npy");     FN_PATH= Path("feature_names_pathway.json")
X_MF  = Path("med_flags_24h.npy"); FN_MF = Path("med_flag_names.json")   # may be (N, K) or (N, 24, K)

Y_FILE= Path("y_labels.npy")

# ---------- Outputs ----------
OUTDIR = Path("models_xgb_cv"); OUTDIR.mkdir(parents=True, exist_ok=True)
OOF_NPY = OUTDIR / "oof_probs.npy"
OOF_CAL = OUTDIR / "oof_probs_calibrated.npy"
OOF_FLD = OUTDIR / "oof_folds.npy"
SUMMARY = OUTDIR / "cv_summary.json"
BAL_IDX = OUTDIR / "balanced_indices.npy"

# ---------- CV / Randomness ----------
N_FOLDS = 5
SEED = 42

# Small, robust grid
GRID = [
    {"max_depth":4, "eta":0.05, "subsample":0.8, "colsample_bytree":0.9, "min_child_weight":1, "lambda":0.0, "alpha":0.0, "tree_method":"hist"},
    {"max_depth":6, "eta":0.05, "subsample":0.8, "colsample_bytree":0.8, "min_child_weight":2, "lambda":0.0, "alpha":0.0, "tree_method":"hist"},
    {"max_depth":6, "eta":0.03, "subsample":0.8, "colsample_bytree":0.8, "min_child_weight":5, "lambda":0.0, "alpha":1.0, "tree_method":"hist"},
]

# ---------- Helpers ----------
def _normalize_block(Xb: np.ndarray, block_label: str) -> np.ndarray:
    """Ensure 2D (N, D). For med flags, pool over time with max; otherwise mean."""
    if Xb.ndim == 1:  return Xb.reshape(-1, 1)
    if Xb.ndim == 2:  return Xb
    if Xb.ndim == 3:
        if block_label == "med":  return np.nanmax(Xb, axis=1)  # any-hour exposure
        else:                     return np.nanmean(Xb, axis=1)  # generic fallback
    raise ValueError(f"Unexpected ndim for {block_label}: {Xb.ndim}")

def _prefix(names, pfx): return [f"{pfx}{n}" for n in names]

def _assert_unique(names):
    seen = {}
    dups = []
    for n in names:
        if n in seen: dups.append(n)
        else: seen[n] = 1
    if dups:
        raise ValueError(f"Duplicate feature names after prefixing: {sorted(set(dups))[:10]} (and possibly more)")

def build_features():
    # base
    X = np.load(X_TAB)
    names = _prefix(json.loads(FN_TAB.read_text()), "tab__")

    # optional scores
    if X_SCO.exists() and FN_SCO.exists():
        Xs = _normalize_block(np.load(X_SCO), "scores")
        if Xs.shape[0] != X.shape[0]:
            print(f"[warn] skipping scores: rows {Xs.shape[0]} != {X.shape[0]}")
        else:
            names_s = _prefix(json.loads(FN_SCO.read_text()), "score__")
            X = np.concatenate([X, Xs], axis=1)
            names += names_s

    # optional pathway
    if X_PATH.exists() and FN_PATH.exists():
        Xp = _normalize_block(np.load(X_PATH), "pathway")
        if Xp.shape[0] != X.shape[0]:
            print(f"[warn] skipping pathway: rows {Xp.shape[0]} != {X.shape[0]}")
        else:
            names_p = _prefix(json.loads(FN_PATH.read_text()), "path__")
            X = np.concatenate([X, Xp], axis=1)
            names += names_p

    # optional med flags
    if X_MF.exists() and FN_MF.exists():
        Xm = _normalize_block(np.load(X_MF), "med")
        if Xm.shape[0] != X.shape[0]:
            print(f"[warn] skipping med flags: rows {Xm.shape[0]} != {X.shape[0]}")
        else:
            names_m = _prefix(json.loads(FN_MF.read_text()), "med__")
            X = np.concatenate([X, Xm], axis=1)
            names += names_m

    if X.shape[1] != len(names):
        raise ValueError(f"Columns ({X.shape[1]}) != names ({len(names)})")
    _assert_unique(names)

    print(f"[features] X shape={X.shape}, names={len(names)}")
    return X, names

def run_fold(X, y, tr, va, feat_names):
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]

    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=feat_names)
    dva = xgb.DMatrix(Xva, label=yva, feature_names=feat_names)

    base = {"objective":"binary:logistic", "eval_metric":"auc", "seed":SEED}
    best = {"auc":-1, "params":None, "booster":None, "iso":None, "best_iter":None}

    for g in GRID:
        params = {**base, **g}
        evallist = [(dtr,"train"), (dva,"val")]
        model = xgb.train(params, dtr, num_boost_round=5000, evals=evallist,
                          early_stopping_rounds=200, verbose_eval=200)
        p_val = model.predict(dva, iteration_range=(0, model.best_iteration+1))
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_val, yva)
        p_cal = iso.transform(p_val)
        auc = roc_auc_score(yva, p_cal)
        if auc > best["auc"]:
            best = {"auc":auc, "params":params, "booster":model,
                    "iso":iso, "best_iter":model.best_iteration}
    return best

def main():
    X, names = build_features()
    y = np.load(Y_FILE).astype(int)

    # Strict 1:1 sampling BEFORE CV
    Xb, yb, kept = make_one_to_one_sample(X, y, random_state=SEED)
    np.save(BAL_IDX, kept)

    oof_raw = np.zeros(len(yb)); oof_cal = np.zeros(len(yb)); oof_fld = np.full(len(yb), -1)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = []

    for k,(tr,va) in enumerate(skf.split(Xb,yb), start=1):
        print(f"\n=== XGB Fold {k}/{N_FOLDS} ===")
        best = run_fold(Xb, yb, tr, va, names)
        best["booster"].save_model(str(OUTDIR / f"xgb_fold{k}.json"))
        joblib.dump(best["iso"], OUTDIR / f"calibrator_fold{k}.pkl")
        folds.append({"fold":k,"auc":best["auc"],"params":best["params"],"best_iter":best["best_iter"]})

        dva = xgb.DMatrix(Xb[va], feature_names=names)
        p = best["booster"].predict(dva, iteration_range=(0, best["best_iter"]+1))
        oof_raw[va] = p
        oof_cal[va] = best["iso"].transform(p)
        oof_fld[va] = k

    oof_auc = roc_auc_score(yb, oof_cal)
    oof_auprc = average_precision_score(yb, oof_cal)
    oof_brier = brier_score_loss(yb, oof_cal)
    print(f"\nXGB OOF (balanced): AUROC={oof_auc:.4f}  AUPRC={oof_auprc:.4f}  Brier={oof_brier:.4f}")

    np.save(OOF_NPY, oof_raw); np.save(OOF_CAL, oof_cal); np.save(OOF_FLD, oof_fld)
    SUMMARY.write_text(json.dumps({
        "oof":{"auroc":oof_auc,"auprc":oof_auprc,"brier":oof_brier},
        "folds":folds,
        "balanced_indices_file": str(BAL_IDX)
    }, indent=2))

if __name__ == "__main__":
    main()
