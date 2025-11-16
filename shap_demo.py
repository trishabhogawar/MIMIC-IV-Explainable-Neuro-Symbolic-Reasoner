# shap_demo.py — SHAP for LightGBM CV fold (feature build mirrors training)
import argparse, json, numpy as np, pandas as pd, shap, lightgbm as lgb
from pathlib import Path
import matplotlib.pyplot as plt

# ---------- Defaults (override via CLI) ----------
MODEL_DIR = Path("models_lgbm_cv")
FOLD      = 1

# Base features
X_TAB = Path("X_tab.npy");            FN_TAB  = Path("feature_names_tab.json")
X_SCO = Path("X_scores.npy");         FN_SCO  = Path("feature_names_scores.json")
X_PATH= Path("X_pathway.npy");        FN_PATH = Path("feature_names_pathway.json")
X_MF  = Path("med_flags_24h.npy");    FN_MF   = Path("med_flag_names.json")  # (N,24,K) or (N,K)

SPLIT  = Path("split_index.csv")  # optional
Y_FILE = Path("y_labels.npy")     # optional

OUTDIR = Path("reports/xai")

# ---------- Helpers ----------
def _prefix(names, pfx): return [f"{pfx}{n}" for n in names]

def _normalize_block(Xb: np.ndarray, label: str) -> np.ndarray:
    """Ensure 2D (N,D). For med flags: pool time with max; otherwise mean over time if 3D."""
    if Xb.ndim == 1:  return Xb.reshape(-1, 1)
    if Xb.ndim == 2:  return Xb
    if Xb.ndim == 3:
        if label == "med":  return np.nanmax(Xb, axis=1)   # any-hour exposure
        else:               return np.nanmean(Xb, axis=1)
    raise ValueError(f"Unexpected ndim for {label}: {Xb.ndim}")

def _assert_unique(names):
    seen=set(); dups=[]
    for n in names:
        if n in seen: dups.append(n)
        seen.add(n)
    if dups:
        raise RuntimeError(f"Duplicate feature names after prefixing, e.g. {sorted(set(dups))[:10]}")

def build_features():
    """Recreate the exact feature matrix used in training (prefix + optional blocks)."""
    assert X_TAB.exists(), "X_tab.npy missing. Run your feature builders first."
    X   = np.load(X_TAB)
    nam = _prefix(json.loads(FN_TAB.read_text()), "tab__")

    if X_SCO.exists() and FN_SCO.exists():
        Xs  = _normalize_block(np.load(X_SCO), "scores")
        nam += _prefix(json.loads(FN_SCO.read_text()), "score__")
        if Xs.shape[0]==X.shape[0]:
            X = np.concatenate([X, Xs], axis=1)
        else:
            print(f"[warn] skipping scores: rows {Xs.shape[0]} != {X.shape[0]}")

    if X_PATH.exists() and FN_PATH.exists():
        Xp  = _normalize_block(np.load(X_PATH), "pathway")
        nam += _prefix(json.loads(FN_PATH.read_text()), "path__")
        if Xp.shape[0]==X.shape[0]:
            X = np.concatenate([X, Xp], axis=1)
        else:
            print(f"[warn] skipping pathway: rows {Xp.shape[0]} != {X.shape[0]}")

    if X_MF.exists() and FN_MF.exists():
        Xm  = _normalize_block(np.load(X_MF), "med")
        nam += _prefix(json.loads(FN_MF.read_text()), "med__")
        if Xm.shape[0]==X.shape[0]:
            X = np.concatenate([X, Xm], axis=1)
        else:
            print(f"[warn] skipping med flags: rows {Xm.shape[0]} != {X.shape[0]}")

    if X.shape[1] != len(nam):
        raise RuntimeError(f"Columns ({X.shape[1]}) != names ({len(nam)}) after build_features")
    _assert_unique(nam)
    return X, nam

def choose_model_and_expected_names(model_dir: Path, fold: int):
    model_path = model_dir / f"lgbm_fold{fold}.txt"
    fn_all     = model_dir / "feature_names_all.json"
    if not model_path.exists() or not fn_all.exists():
        raise FileNotFoundError(f"Missing model or names: {model_path}, {fn_all}. Train CV first.")
    booster = lgb.Booster(model_file=str(model_path))
    names_expected = json.loads(fn_all.read_text())
    return booster, names_expected

def align_to_model(X, names, names_expected):
    """Reorder columns to match trained model’s feature order."""
    if len(names) != len(names_expected):
        sa, sb = set(names), set(names_expected)
        missing = [n for n in names_expected if n not in sa]
        extra   = [n for n in names if n not in sb]
        raise RuntimeError(
            f"Feature mismatch: X={len(names)} vs model={len(names_expected)}.\n"
            f"Missing (first 10): {missing[:10]}\nExtra (first 10): {extra[:10]}"
        )
    pos = {n:i for i,n in enumerate(names)}
    order = [pos[n] for n in names_expected]
    return X[:, order], names_expected

def subset_rows(X, model_dir: Path, limit: int|None, use_balanced: bool):
    """Optionally use training balanced indices and/or limit for speed."""
    idx = None
    if use_balanced:
        bpath = model_dir / "balanced_indices.npy"
        if not bpath.exists():
            raise FileNotFoundError(f"{bpath} not found. Re-train or disable --use-balanced.")
        idx = np.load(bpath).astype(int)
        X = X[idx]
    if limit is not None and X.shape[0] > limit:
        X = X[:limit]
        if idx is not None:
            idx = idx[:limit]
    return X, idx

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(MODEL_DIR), help="CV model directory")
    ap.add_argument("--fold", type=int, default=FOLD, help="Fold number to explain")
    ap.add_argument("--limit", type=int, default=5000, help="Limit rows for SHAP (memory/time)")
    ap.add_argument("--use-balanced", action="store_true",
                    help="Use training balanced subset (balanced_indices.npy) for SHAP")
    args = ap.parse_args()

    outdir = OUTDIR; outdir.mkdir(parents=True, exist_ok=True)

    # 1) Build features like training
    X, names = build_features()

    # 2) Load model + expected names; align order
    booster, names_expected = choose_model_and_expected_names(Path(args.model_dir), args.fold)
    X, names_aligned = align_to_model(X, names, names_expected)

    # 3) Optional: use the balanced subset &/or limit rows
    X_plot, kept = subset_rows(X, Path(args.model_dir), args.limit, args.use_balanced)
    print(f"[shap] Using {X_plot.shape[0]} rows for SHAP (aligned D={X_plot.shape[1]})")

    # 4) SHAP
    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(X_plot)

    # 5) Plots
    # Beeswarm
    shap.summary_plot(sv, X_plot, feature_names=names_aligned, show=False)
    plt.tight_layout(); plt.savefig(outdir / f"lgbm_fold{args.fold}_shap_summary.png", dpi=180); plt.close()

    # Bar
    shap.summary_plot(sv, X_plot, feature_names=names_aligned, plot_type="bar", show=False)
    plt.tight_layout(); plt.savefig(outdir / f"lgbm_fold{args.fold}_shap_bar.png", dpi=180); plt.close()

    # 6) Save the used indices if balanced or limited
    if kept is not None:
        np.save(outdir / f"shap_rows_indices_fold{args.fold}.npy", kept)

    print(f"[shap] Saved plots in {outdir}")

if __name__ == "__main__":
    main()
