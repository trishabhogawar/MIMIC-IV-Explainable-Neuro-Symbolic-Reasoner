# ensemble_blend.py
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def load_oof(model_dir: Path):
    p = np.load(model_dir / "oof_probs_calibrated.npy").astype(float).ravel()
    idx_path = model_dir / "balanced_indices.npy"
    if not idx_path.exists():
        raise FileNotFoundError(f"{idx_path} missing. Your training used 1:1 sampling; "
                                f"please save balanced_indices.npy with the model.")
    idx = np.load(idx_path).astype(int).ravel()
    if len(p) != len(idx):
        raise ValueError(f"length mismatch in {model_dir}: probs={len(p)} vs indices={len(idx)}")
    return p, idx

def align_on_intersection(p1, i1, p2, i2):
    # intersection of kept indices (sorted)
    common = np.intersect1d(i1, i2, assume_unique=False)
    if len(common) == 0:
        raise ValueError("No overlap between balanced indices of the two models.")
    # map original index->position for fast gather
    pos1 = {int(a): j for j, a in enumerate(i1)}
    pos2 = {int(a): j for j, a in enumerate(i2)}
    a1 = np.array([p1[pos1[int(k)]] for k in common], dtype=float)
    a2 = np.array([p2[pos2[int(k)]] for k in common], dtype=float)
    return a1, a2, common

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lgbm", default="models_lgbm_cv", help="dir with LGBM OOF + balanced_indices.npy")
    ap.add_argument("--xgb",  default="models_xgb_cv",  help="dir with XGB  OOF + balanced_indices.npy")
    ap.add_argument("--labels", default="y_labels.npy", help="full-cohort labels npy")
    ap.add_argument("--out", default="models_blend_cv", help="output dir")
    ap.add_argument("--weight", type=float, default=0.5, help="blend weight for LGBM (xgb uses 1-w)")
    args = ap.parse_args()

    d_lgb = Path(args.lgbm); d_xgb = Path(args.xgb); outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # load
    p_lgb, idx_lgb = load_oof(d_lgb)
    p_xgb, idx_xgb = load_oof(d_xgb)
    y_full = np.load(args.labels).astype(int)

    # align
    if np.array_equal(idx_lgb, idx_xgb):
        kept = idx_lgb
        a_lgb, a_xgb = p_lgb, p_xgb
        print("[blend] Balanced indices match exactly.")
    else:
        a_lgb, a_xgb, kept = align_on_intersection(p_lgb, idx_lgb, p_xgb, idx_xgb)
        print(f"[blend] Indices differ; using intersection of size {len(kept)} "
              f"(LGBM={len(idx_lgb)}, XGB={len(idx_xgb)}).")

    # labels on aligned subset
    y = y_full[kept]

    # blend
    w = args.weight
    p_blend = w * a_lgb + (1.0 - w) * a_xgb

    # metrics
    auroc = roc_auc_score(y, p_blend)
    auprc = average_precision_score(y, p_blend)
    brier = brier_score_loss(y, p_blend)
    print(f"BLEND OOF (w={w:.2f} on LGBM): AUROC={auroc:.4f}  AUPRC={auprc:.4f}  Brier={brier:.4f}")

    # save artifacts
    np.save(outdir / "oof_probs_calibrated.npy", p_blend)
    np.save(outdir / "balanced_indices.npy", kept)
    (outdir / "blend_info.txt").write_text(
        f"source_lgbm={d_lgb}\nsource_xgb={d_xgb}\nweight_lgbm={w}\n"
        f"n_common={len(kept)}\n"
        f"auroc={auroc:.6f}\nauprc={auprc:.6f}\nbrier={brier:.6f}\n"
    )
    print(f"[blend] Saved blended OOF and indices to {outdir}")

if __name__ == "__main__":
    main()
