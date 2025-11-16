# make_med_flags.py
import pandas as pd
import numpy as np
from pathlib import Path
import json

# ---------- I/O ----------
MED_EVENTS_CSV = Path("meds_24h.csv")   # your extracted prescriptions/exposures within first 24h
COHORT_CSV     = Path("cohort_index.csv")

OUT_DAY_FLAGS  = Path("med_day_flags.csv")   # per-day 0/1 flags (N x K) with stay_id
OUT_HOUR_FLAGS = Path("med_flags_24h.npy")   # hourly 0/1 flags (N x 24 x K)
OUT_NAMES      = Path("med_flag_names.json")

# ---------- Med class dictionaries ----------
CLASS_MAP = {
    "pressor":       ["norepinephrine","epinephrine","dopamine","phenylephrine","vasopressin"],
    "sedation":      ["propofol","midazolam","fentanyl","dexmedetomidine"],
    "beta_blocker":  ["metoprolol","esmolol","propranolol","atenolol"],
}
K_ORDER = ["pressor","sedation","beta_blocker"]

# ---------- Helpers ----------
def norm(s) -> str:
    return str(s).strip().lower()

def clamp_hour(h):
    try:
        h = int(h)
    except Exception:
        return None
    return max(0, min(23, h))

def rows_to_hours(row, intime_by_stay):
    """
    Return (hs, he) as inclusive hour indices [0..23] for a given row.
    Priority:
      1) hour_start/hour_end columns
      2) hour column (single hour)
      3) starttime/endtime timestamps -> compute rel hours using cohort intime
    Fallback: None (caller will skip)
    """
    # 1) Explicit hours
    if "hour_start" in row and "hour_end" in row and pd.notna(row["hour_start"]) and pd.notna(row["hour_end"]):
        hs = clamp_hour(row["hour_start"])
        he = clamp_hour(row["hour_end"])
        if hs is None or he is None: return None
        if he < hs: hs, he = he, hs
        return hs, he

    # 2) Single hour
    if "hour" in row and pd.notna(row["hour"]):
        h = clamp_hour(row["hour"])
        if h is None: return None
        return h, h

    # 3) Timestamps
    sid = row.get("stay_id", None)
    if sid is not None and sid in intime_by_stay:
        it = intime_by_stay[sid]
        # starttime/endtime OR charttime columns
        st = row.get("starttime", row.get("charttime"))
        et = row.get("endtime",   row.get("charttime"))
        try:
            st = pd.to_datetime(st) if pd.notna(st) else None
            et = pd.to_datetime(et) if pd.notna(et) else None
        except Exception:
            st, et = None, None

        if st is not None:
            hs = clamp_hour(np.floor((st - it).total_seconds() / 3600.0))
        else:
            hs = 0
        if et is not None:
            he = clamp_hour(np.floor((et - it).total_seconds() / 3600.0))
        else:
            he = hs

        if hs is None or he is None: return None
        if he < hs: hs, he = he, hs
        return hs, he

    return None

def main():
    # ---------- Load cohort (source of truth for N and order) ----------
    if not COHORT_CSV.exists():
        raise FileNotFoundError(f"{COHORT_CSV} not found. Build it via build_day1_dataset.py first.")
    cohort = pd.read_csv(COHORT_CSV)
    if "stay_id" not in cohort.columns:
        raise RuntimeError("cohort_index.csv must contain stay_id.")
    N = len(cohort)
    stay_ids = cohort["stay_id"].astype(int).tolist()
    intime_by_stay = dict(zip(cohort["stay_id"].astype(int), pd.to_datetime(cohort["intime"])))

    # ---------- Load med events ----------
    if not MED_EVENTS_CSV.exists():
        raise FileNotFoundError(f"{MED_EVENTS_CSV} not found. Produce it from extract_meds_24h.py.")
    meds = pd.read_csv(MED_EVENTS_CSV)

    # Ensure stay_id exists (if only subject_id present, we can’t align per-stay reliably)
    if "stay_id" not in meds.columns:
        raise RuntimeError("meds_24h.csv must contain stay_id for alignment.")

    # Normalize drug name column
    if "drug_text" in meds.columns:
        name_col = "drug_text"
    elif "drug" in meds.columns:
        name_col = "drug"
    else:
        # last resort: first textual column
        text_cols = [c for c in meds.columns if meds[c].dtype == object]
        name_col = text_cols[0] if text_cols else meds.columns[-1]
    meds[name_col] = meds[name_col].map(norm)

    # Keep only cohort stays; drop obvious dup rows
    meds = meds[meds["stay_id"].isin(stay_ids)].copy()
    meds = meds.drop_duplicates()

    # ---------- Build hourly flag tensor ----------
    K = len(K_ORDER); T = 24
    H = np.zeros((N, T, K), dtype=np.uint8)
    row_index = {sid: i for i, sid in enumerate(stay_ids)}

    # Iterate meds
    vocab_map = {c: [norm(x) for x in vs] for c, vs in CLASS_MAP.items()}

    # Pre-vectorize for speed: for each class, match rows containing any vocab string
    for _, r in meds.iterrows():
        sid = r.get("stay_id", None)
        if pd.isna(sid): 
            continue
        sid = int(sid)
        i = row_index.get(sid, None)
        if i is None:
            continue

        nm = r[name_col]
        if not isinstance(nm, str) or not nm:
            continue

        hrs = rows_to_hours(r, intime_by_stay)
        if hrs is None:
            continue
        hs, he = hrs

        for cls_idx, cls in enumerate(K_ORDER):
            if any(v in nm for v in vocab_map[cls]):
                H[i, hs:he+1, cls_idx] = 1

    # ---------- Per-day collapsed CSV ----------
    day_flags = (H.sum(axis=1) > 0).astype(int)   # (N, K)
    df = cohort[["stay_id"]].copy()
    for j, cls in enumerate(K_ORDER):
        df[f"day_{cls}"] = day_flags[:, j]
    df.to_csv(OUT_DAY_FLAGS, index=False)

    # ---------- Hourly npy + names ----------
    np.save(OUT_HOUR_FLAGS, H)
    OUT_NAMES.write_text(json.dumps(K_ORDER, indent=2))

    print(f"[med_flags] Saved {OUT_DAY_FLAGS}  shape={(N, len(K_ORDER))}")
    print(f"[med_flags] Saved {OUT_HOUR_FLAGS} shape={H.shape}")
    print(f"[med_flags] Saved {OUT_NAMES}")

if __name__ == "__main__":
    main()
