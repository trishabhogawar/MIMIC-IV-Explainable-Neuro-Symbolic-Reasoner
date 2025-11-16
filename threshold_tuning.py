# threshold_tuning.py
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_curve

VAL_LOGITS = Path("reports/dl/val_logits.npy")   # save these from your trainer if needed
VAL_LABELS = Path("reports/dl/val_labels.npy")
OUT_CSV    = Path("reports/metrics/threshold_sweep.csv")

def main():
    y = np.load(VAL_LABELS); z = np.load(VAL_LOGITS)
    p = 1/(1+np.exp(-z))
    prec, rec, thr = precision_recall_curve(y, p)
    pd.DataFrame({"threshold": np.r_[thr, 1.0], "precision": prec, "recall": rec}).to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")

if __name__ == "__main__":
    main()
