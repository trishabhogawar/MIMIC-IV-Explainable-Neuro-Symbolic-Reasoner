# evaluate_oof.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, f1_score
)

from evaluation_metrics import expected_calibration_error, decision_curve


def pick_thresholds(y_true: np.ndarray, p: np.ndarray):
    """Return default 0.5, best F1, and a high-sensitivity (recall~0.85) threshold."""
    th_default = 0.5

    precision, recall, ths_pr = precision_recall_curve(y_true, p)
    f1s = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    ix_f1 = int(np.nanargmax(f1s)) if len(f1s) > 0 else 0
    th_best_f1 = float(ths_pr[ix_f1]) if len(ths_pr) else th_default

    target_recall = 0.85
    candidates = np.where(recall[:-1] >= target_recall)[0]
    th_high_sens = float(ths_pr[candidates[0]]) if len(candidates) else th_default

    return {
        "default_0.5": th_default,
        "best_f1": th_best_f1,
        "high_sensitivity": th_high_sens,
    }


def print_and_save_clf_report(y_true: np.ndarray, p: np.ndarray, thresholds: dict, outdir: Path):
    rows = []
    for name, th in thresholds.items():
        y_pred = (p >= th).astype(int)
        rep = classification_report(
            y_true, y_pred, target_names=["Survived", "Died"], digits=4
        )
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"\nClassification Report @ {name} (threshold={th:.4f})")
        print(rep)
        print(f"Confusion Matrix @ {name}:\nTN={tn}  FP={fp}  FN={fn}  TP={tp}")

        rows.append({
            "threshold_name": name,
            "threshold": th,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "F1": f1_score(y_true, y_pred)
        })

        (outdir / f"classification_report_{name}.txt").write_text(rep)
        pd.DataFrame(cm, index=["True 0","True 1"], columns=["Pred 0","Pred 1"])\
          .to_csv(outdir / f"confusion_matrix_{name}.csv")

    pd.DataFrame(rows).to_csv(outdir / "thresholds_summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models_lgbm_cv",
                    help="Directory containing oof_probs_calibrated.npy (and balanced_indices.npy)")
    ap.add_argument("--y", default="y_labels.npy", help="Path to labels npy (full cohort)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    outdir = Path("reports") / ("cv_eval_" + model_dir.name)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load y (full cohort) ---
    y_full = np.load(args.y).astype(int)

    # --- Load OOF calibrated probs ---
    p_path = model_dir / "oof_probs_calibrated.npy"
    if not p_path.exists():
        raise FileNotFoundError(f"Missing {p_path}")
    p = np.load(p_path).astype(float).ravel()

    # --- If 1:1 sampling was used, align y using the saved indices ---
    idx_path = model_dir / "balanced_indices.npy"
    if idx_path.exists():
        kept = np.load(idx_path).astype(int)
        y = y_full[kept]
        print(f"[evaluate] Using balanced subset: y={len(y)} to match p={len(p)}")
    else:
        y = y_full
        print(f"[evaluate] No balanced_indices.npy; using full y={len(y)}")

    if len(y) != len(p):
        raise ValueError(f"Length mismatch: y={len(y)} vs p={len(p)}. "
                         f"Check that {idx_path} matches the OOF file.")

    # --- Threshold-independent metrics ---
    auroc = roc_auc_score(y, p)
    auprc = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    ece = expected_calibration_error(y, p)
    print(f"OOF: AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Brier={brier:.4f}  ECE={ece:.4f}")

    pd.Series({"AUROC": auroc, "AUPRC": auprc, "Brier": brier, "ECE": ece})\
      .to_csv(outdir / "oof_summary.csv")

    # --- Classification report(s) ---
    thresholds = pick_thresholds(y, p)
    print_and_save_clf_report(y, p, thresholds, outdir)

    # --- Calibration plot ---
    bins = np.linspace(0, 1, 11); mids = 0.5 * (bins[1:] + bins[:-1])
    d = np.digitize(p, bins) - 1
    obs = [y[d == i].mean() if np.any(d == i) else np.nan for i in range(len(mids))]
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "--", lw=1, label="Perfect")
    ax.plot(mids, obs, marker="o", label="OOF")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Observed"); ax.set_title("OOF Calibration")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(outdir / "oof_calibration.png", dpi=160); plt.close(fig)

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC (AUC={auroc:.3f})")
    ax.plot([0,1],[0,1],"--", lw=1, color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("OOF ROC"); ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "oof_roc.png", dpi=160); plt.close(fig)

    # --- PR curve ---
    prec, rec, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots()
    ax.plot(rec, prec, label=f"PR (AP={auprc:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("OOF Precision–Recall"); ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "oof_pr.png", dpi=160); plt.close(fig)

    # --- Decision curve ---
    dca = decision_curve(y, p)
    dca.to_csv(outdir / "oof_decision_curve.csv", index=False)

    print(f"[evaluate] Wrote reports to {outdir}")

if __name__ == "__main__":
    main()
