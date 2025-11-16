# train_lgbm_tabular_cv.py
import json, numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import joblib
from utils_sampling import make_one_to_one_sample  # you already have this

# core tabular
X_TAB = Path("X_tab.npy")
FN_TAB= Path("feature_names_tab.json")

# optional blocks
X_SCO = Path("X_scores.npy");      FN_SCO= Path("feature_names_scores.json")
X_PATH= Path("X_pathway.npy");     FN_PATH= Path("feature_names_pathway.json")
X_MF  = Path("med_flags_24h.npy"); FN_MF = Path("med_flag_names.json")   # (N,24,K) or (N,K)

Y_FILE= Path("y_labels.npy")

OUTDIR = Path("models_lgbm_cv"); OUTDIR.mkdir(parents=True, exist_ok=True)
OOF_NPY = OUTDIR / "oof_probs.npy"
OOF_CAL = OUTDIR / "oof_probs_calibrated.npy"
OOF_FLD = OUTDIR / "oof_folds.npy"
FOLD_P  = OUTDIR / "fold_models.txt"
SUMMARY = OUTDIR / "cv_summary.json"
BAL_IDX = OUTDIR / "balanced_indices.npy"

N_FOLDS = 5
SEED = 42

PARAM_GRID = [
    {"num_leaves": 31, "max_depth": 6,  "min_data_in_leaf": 60,  "feature_fraction": 0.9,  "lambda_l1":0.0, "lambda_l2":0.0},
    {"num_leaves": 63, "max_depth": 8,  "min_data_in_leaf": 100, "feature_fraction": 0.8,  "lambda_l1":0.0, "lambda_l2":0.0},
    {"num_leaves": 63, "max_depth": -1, "min_data_in_leaf": 120, "feature_fraction": 0.75, "lambda_l1":0.0, "lambda_l2":0.0},
    {"num_leaves": 127,"max_depth": -1, "min_data_in_leaf": 200, "feature_fraction": 0.7,  "lambda_l1":0.0, "lambda_l2":0.0},
    {"num_leaves": 31, "max_depth": 6,  "min_data_in_leaf": 80,  "feature_fraction": 0.9,  "lambda_l1":0.0, "lambda_l2":1.0},
]

def _normalize_block(Xb: np.ndarray, block_label: str) -> np.ndarray:
    """Ensure 2D (N, D). For med flags, pool over time with max."""
    if Xb.ndim == 1:  return Xb.reshape(-1, 1)
    if Xb.ndim == 2:  return Xb
    if Xb.ndim == 3:  # time dimension at axis=1
        if block_label == "med":  return np.nanmax(Xb, axis=1)
        else:                     return np.nanmean(Xb, axis=1)
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
    X = np.load(X_TAB)
    names = _prefix(json.loads(FN_TAB.read_text()), "tab__")

    if X_SCO.exists() and FN_SCO.exists():
        Xs = _normalize_block(np.load(X_SCO), "scores")
        if Xs.shape[0] != X.shape[0]:
            print(f"[warn] skipping scores: rows {Xs.shape[0]} != {X.shape[0]}")
        else:
            names_s = _prefix(json.loads(FN_SCO.read_text()), "score__")
            X = np.concatenate([X, Xs], axis=1)
            names += names_s

    if X_PATH.exists() and FN_PATH.exists():
        Xp = _normalize_block(np.load(X_PATH), "pathway")
        if Xp.shape[0] != X.shape[0]:
            print(f"[warn] skipping pathway: rows {Xp.shape[0]} != {X.shape[0]}")
        else:
            names_p = _prefix(json.loads(FN_PATH.read_text()), "path__")
            X = np.concatenate([X, Xp], axis=1)
            names += names_p

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

def run_fold(X, y, train_idx, val_idx, feat_names):
    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[val_idx], y[val_idx]

    base = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
    }

    best = {"auc": -1, "params": None, "booster": None, "iter": None, "iso": None}
    dtr = lgb.Dataset(Xtr, ytr, feature_name=feat_names)
    dva = lgb.Dataset(Xva, yva, feature_name=feat_names, reference=dtr)

    for g in PARAM_GRID:
        params = {**base, **g}
        model = lgb.train(
            params, dtr, valid_sets=[dtr, dva], valid_names=["train","val"],
            num_boost_round=5000,
            callbacks=[lgb.early_stopping(stopping_rounds=200), lgb.log_evaluation(period=200)]
        )
        p_val_raw = model.predict(Xva, num_iteration=model.best_iteration)
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_val_raw, yva)
        p_cal = iso.transform(p_val_raw)
        auc = roc_auc_score(yva, p_cal)
        if auc > best["auc"]:
            best = {"auc": auc, "params": params, "booster": model,
                    "iter": model.best_iteration, "iso": iso}
    return best

def main():
    X, feat_names = build_features()
    y = np.load(Y_FILE).astype(int)

    # strict 1:1 sampling BEFORE CV
    Xb, yb, kept = make_one_to_one_sample(X, y, random_state=SEED)
    np.save(BAL_IDX, kept)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_raw = np.zeros(len(yb), dtype=float)
    oof_cal = np.zeros(len(yb), dtype=float)
    oof_fld = np.full(len(yb), -1, dtype=int)

    fold_params = []
    for k, (tr, va) in enumerate(skf.split(Xb, yb), start=1):
        print(f"\n=== Fold {k}/{N_FOLDS} ===")
        best = run_fold(Xb, yb, tr, va, feat_names)

        OUTDIR.mkdir(parents=True, exist_ok=True)
        best["booster"].save_model(str(OUTDIR / f"lgbm_fold{k}.txt"))
        joblib.dump(best["iso"], OUTDIR / f"calibrator_fold{k}.pkl")
        fold_params.append({"fold": k, "auc": best["auc"],
                            "params": best["params"], "best_iter": best["iter"]})

        p_raw = best["booster"].predict(Xb[va], num_iteration=best["iter"])
        p_cal = best["iso"].transform(p_raw)
        oof_raw[va] = p_raw
        oof_cal[va] = p_cal
        oof_fld[va] = k

    oof_auc = roc_auc_score(yb, oof_cal)
    oof_auprc = average_precision_score(yb, oof_cal)
    oof_brier = brier_score_loss(yb, oof_cal)
    print(f"\nOOF (balanced): AUROC={oof_auc:.4f}  AUPRC={oof_auprc:.4f}  Brier={oof_brier:.4f}")

    np.save(OOF_NPY, oof_raw); np.save(OOF_CAL, oof_cal); np.save(OOF_FLD, oof_fld)
    (OUTDIR / "feature_names_all.json").write_text(json.dumps(feat_names, indent=2))
    SUMMARY.write_text(json.dumps({
        "oof": {"auroc": oof_auc, "auprc": oof_auprc, "brier": oof_brier},
        "folds": fold_params,
        "balanced_indices_file": str(BAL_IDX)
    }, indent=2))
    FOLD_P.write_text("\n".join([str(p) for p in fold_params]))

if __name__ == "__main__":
    main()
