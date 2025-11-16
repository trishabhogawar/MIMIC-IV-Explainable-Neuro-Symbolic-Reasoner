# evaluation_metrics.py
import argparse, json
from pathlib import Path
import numpy as np, matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve
)

# ----------------- Feature rebuild (mirrors trainer) -----------------
X_TAB = Path("X_tab.npy");            FN_TAB  = Path("feature_names_tab.json")
X_SCO = Path("X_scores.npy");         FN_SCO  = Path("feature_names_scores.json")
X_PATH= Path("X_pathway.npy");        FN_PATH = Path("feature_names_pathway.json")
X_MF  = Path("med_flags_24h.npy");    FN_MF   = Path("med_flag_names.json")

def _prefix(names, pfx): return [f"{pfx}{n}" for n in names]

def _normalize_block(Xb: np.ndarray, label: str) -> np.ndarray:
    if Xb.ndim == 1:  return Xb.reshape(-1, 1)
    if Xb.ndim == 2:  return Xb
    if Xb.ndim == 3:
        # Pool time: meds=any-hour (max), others mean
        return np.nanmax(Xb, axis=1) if label == "med" else np.nanmean(Xb, axis=1)
    raise ValueError(f"Unexpected ndim for {label}: {Xb.ndim}")

def build_features_like_training():
    assert X_TAB.exists(), "X_tab.npy missing."
    import json
    X   = np.load(X_TAB)
    nam = _prefix(json.loads(FN_TAB.read_text()), "tab__")

    if X_SCO.exists() and FN_SCO.exists():
        Xs = _normalize_block(np.load(X_SCO), "scores")
        if Xs.shape[0] == X.shape[0]:
            X = np.concatenate([X, Xs], axis=1)
            nam += _prefix(json.loads(FN_SCO.read_text()), "score__")

    if X_PATH.exists() and FN_PATH.exists():
        Xp = _normalize_block(np.load(X_PATH), "pathway")
        if Xp.shape[0] == X.shape[0]:
            X = np.concatenate([X, Xp], axis=1)
            nam += _prefix(json.loads(FN_PATH.read_text()), "path__")

    if X_MF.exists() and FN_MF.exists():
        Xm = _normalize_block(np.load(X_MF), "med")
        if Xm.shape[0] == X.shape[0]:
            X = np.concatenate([X, Xm], axis=1)
            nam += _prefix(json.loads(FN_MF.read_text()), "med__")

    if X.shape[1] != len(nam):
        raise RuntimeError(f"Columns ({X.shape[1]}) != names ({len(nam)})")
    return X, nam

# ----------------- Plots -----------------
def plot_roc(y, p, outdir, title):
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(outdir / "oof_roc.png", dpi=160); plt.close()
    return auc

def plot_pr(y, p, outdir, title):
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    plt.figure()
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(outdir / "oof_pr.png", dpi=160); plt.close()
    return ap

def plot_calibration(y, p, outdir, title):
    """
    Figure 6: Calibration curve comparing predicted vs. observed mortality risk.
    Saves: oof_calibration.png and figure_6_calibration.png
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import brier_score_loss

    # bins & observed rates
    bins = np.linspace(0, 1, 11)
    mids = 0.5 * (bins[1:] + bins[:-1])
    d = np.digitize(p, bins) - 1
    obs = [y[d == i].mean() if np.any(d == i) else np.nan for i in range(len(mids))]
    cnt = [(d == i).sum() for i in range(len(mids))]
    brier = brier_score_loss(y, p)

    # plot
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(mids, obs, "o-", label="Observed (binned)")
    # optionally annotate bin counts
    for xi, yi, n in zip(mids, obs, cnt):
        if np.isfinite(yi):
            ax.annotate(str(int(n)), (xi, yi), textcoords="offset points", xytext=(0,6), ha="center", fontsize=8)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted mortality risk")
    ax.set_ylabel("Observed mortality frequency")
    ax.set_title("Figure 6: Calibration curve comparing predicted vs. observed mortality risk")
    ax.legend(loc="lower right")
    fig.tight_layout()

    # save with two names
    fig.savefig(outdir / "oof_calibration.png", dpi=160)
    fig.savefig(outdir / "figure_6_calibration.png", dpi=160)
    plt.close(fig)
    return brier


def plot_train_vs_val_auc_per_fold(model_dir: Path, outdir: Path, use_balanced: bool):
    """
    Compute AUROC on each fold's train vs val by reloading fold models and rebuilding features.
    """
    import lightgbm as lgb, joblib

    # Load oof fold labels (for val indices)
    folds_path = model_dir / "oof_folds.npy"
    if not folds_path.exists():
        print("[compare] oof_folds.npy not found; skipping train-vs-val graph.")
        return
    fold_lab = np.load(folds_path).astype(int)

    # Load OOF-calibrated probs (for sanity/val check)
    p_oof = np.load(model_dir / "oof_probs_calibrated.npy").astype(float)

    # Balanced indices alignment
    y_full = np.load("y_labels.npy").astype(int)
    kept = None
    if use_balanced:
        bpath = model_dir / "balanced_indices.npy"
        if not bpath.exists():
            print("[compare] balanced_indices.npy missing; skipping train-vs-val graph.")
            return
        kept = np.load(bpath).astype(int)
        y_sub = y_full[kept]
    else:
        y_sub = y_full

    # Build features and align to model feature order
    X, names = build_features_like_training()
    if kept is not None: X = X[kept]

    fn_all = model_dir / "feature_names_all.json"
    if not fn_all.exists():
        print("[compare] feature_names_all.json missing; skipping train-vs-val graph.")
        return
    names_expected = json.loads(fn_all.read_text())
    pos = {n:i for i,n in enumerate(names)}
    try:
        order = [pos[n] for n in names_expected]
    except KeyError as e:
        print(f"[compare] feature mismatch {e}; skipping.")
        return
    X = X[:, order]

    # For each fold: train idx = not this fold; val idx = this fold
    # Load model + calibrator → predict train & val → AUROC
    fold_aucs_train, fold_aucs_val = [], []
    unique_folds = sorted([f for f in np.unique(fold_lab) if f != -1])
    for f in unique_folds:
        tr = np.where(fold_lab != f)[0]
        va = np.where(fold_lab == f)[0]

        model_path = model_dir / f"lgbm_fold{f}.txt"
        iso_path   = model_dir / f"calibrator_fold{f}.pkl"
        if not model_path.exists() or not iso_path.exists():
            print(f"[compare] Missing artifacts for fold {f}; skipping.")
            continue
        booster = lgb.Booster(model_file=str(model_path))
        iso     = joblib.load(iso_path)

        p_tr = iso.transform(booster.predict(X[tr]))
        p_va = iso.transform(booster.predict(X[va]))

        auc_tr = roc_auc_score(y_sub[tr], p_tr)
        auc_va = roc_auc_score(y_sub[va], p_va)
        fold_aucs_train.append(auc_tr)
        fold_aucs_val.append(auc_va)

    if fold_aucs_train and fold_aucs_val:
        # Bar chart
        idx = np.arange(len(fold_aucs_train))
        width = 0.35
        plt.figure()
        plt.bar(idx - width/2, fold_aucs_train, width, label="Train AUROC")
        plt.bar(idx + width/2, fold_aucs_val,   width, label="Val AUROC (OOF)")
        plt.xticks(idx, [f"F{f}" for f in unique_folds])
        plt.ylim(0.5, 1.0)
        plt.ylabel("AUROC"); plt.title("Train vs Validation AUROC per Fold")
        plt.legend(); plt.tight_layout()
        plt.savefig(outdir / "train_vs_val_fold_auc.png", dpi=160); plt.close()
        print("[compare] Saved train_vs_val_fold_auc.png")

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models_lgbm_cv",
                    help="Folder with oof_probs_calibrated.npy and (optional) balanced_indices.npy")
    ap.add_argument("--y", default="y_labels.npy", help="Full-cohort labels file")
    ap.add_argument("--compare-train-val", action="store_true",
                    help="Recompute AUROC on each fold's train vs val and plot comparison")
    args = ap.parse_args()

    mdir = Path(args.model_dir)
    outdir = Path("reports") / ("cv_eval_" + mdir.name)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load predictions and align labels
    p = np.load(mdir / "oof_probs_calibrated.npy").astype(float).ravel()
    y_full = np.load(args.y).astype(int)
    idx_path = mdir / "balanced_indices.npy"
    if idx_path.exists():
        kept = np.load(idx_path).astype(int)
        y = y_full[kept]
        print(f"[eval] Using balanced subset: y={len(y)} matches p={len(p)}")
    else:
        y = y_full
        print(f"[eval] No balanced_indices.npy; using full y={len(y)}")

    if len(y) != len(p):
        raise ValueError(f"Length mismatch: y={len(y)} vs p={len(p)}. "
                         f"Use the same model-dir that produced the OOF probs.")

    # Scalar metrics
    auroc = roc_auc_score(y, p)
    auprc = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    print(f"OOF: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Brier={brier:.4f}")

    # Plots
    plot_roc(y, p, outdir, "OOF ROC")
    plot_pr(y, p, outdir, "OOF Precision–Recall")
    plot_calibration(y, p, outdir, "OOF Calibration")

    # Optional: train vs val comparison across folds
    if args.compare_train_val:
        plot_train_vs_val_auc_per_fold(mdir, outdir, use_balanced=idx_path.exists())

    print(f"[eval] Wrote plots to {outdir}")

if __name__ == "__main__":
    main()
