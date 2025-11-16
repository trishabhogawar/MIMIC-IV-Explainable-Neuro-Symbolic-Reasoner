# extract_meds_24h.py
"""
Create meds_24h.csv aligned to cohort_index.csv
- Reads cohort_index.csv to get (stay_id, hadm_id, intime)
- Extracts hospital prescriptions that overlap the first 24h after ICU intime
- Outputs per-row intervals in hours: [hour_start, hour_end] in [0..23]
Columns: stay_id, drug_text, hour_start, hour_end
"""

import os, getpass
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine, text

COHORT = Path("cohort_index.csv")
OUTCSV = Path("meds_24h.csv")
HOURS  = 24

# DB engine (same convention as other scripts)
DB_USER = os.environ.get("PGUSER") or getpass.getuser()
engine  = create_engine(f"postgresql+psycopg2://{DB_USER}@localhost:5432/mimiciv", future=True)

def clamp_hour(x):
    if pd.isna(x): return None
    try:
        x = int(np.floor(x))
    except Exception:
        return None
    return max(0, min(23, x))

def main():
    if not COHORT.exists():
        raise FileNotFoundError("cohort_index.csv not found. Build it via build_day1_dataset.py first.")
    cohort = pd.read_csv(COHORT)
    req = {"stay_id","hadm_id","intime"}
    if not req.issubset(cohort.columns):
        raise RuntimeError(f"cohort_index.csv must have columns {req}")

    cohort["intime"] = pd.to_datetime(cohort["intime"])
    stays = cohort[["stay_id","hadm_id","intime"]].drop_duplicates()
    print(f"[extract_meds] Cohort rows: {len(stays)}")

    # Pull prescriptions for these admissions (hadm_id) with their drug names and times
    hadm_ids = stays["hadm_id"].astype(int).unique().tolist()
    # Split into chunks to avoid SQL parameter length limits if huge
    CHUNK = 5000
    rows = []
    with engine.connect() as conn:
        for i in range(0, len(hadm_ids), CHUNK):
            chunk = hadm_ids[i:i+CHUNK]
            sql = text("""
                SELECT
                    p.hadm_id,
                    p.starttime,
                    p.stoptime,
                    COALESCE(NULLIF(TRIM(p.drug_name_generic), ''), NULLIF(TRIM(p.drug), '')) AS drug_text
                FROM mimiciv_hosp.prescriptions p
                WHERE p.hadm_id = ANY(:hadms)
            """)
            df = pd.read_sql(sql, conn, params={"hadms": chunk})
            rows.append(df)
    if rows:
        presc = pd.concat(rows, ignore_index=True)
    else:
        presc = pd.DataFrame(columns=["hadm_id","starttime","stoptime","drug_text"])

    # Clean times and drug names
    presc["starttime"] = pd.to_datetime(presc["starttime"], errors="coerce")
    presc["stoptime"]  = pd.to_datetime(presc["stoptime"],  errors="coerce")
    presc["drug_text"] = presc["drug_text"].astype(str).str.strip()
    presc = presc[~presc["drug_text"].eq("")].copy()

    # Join to ICU stay_id & intime
    j = presc.merge(stays, on="hadm_id", how="inner")

    # Compute relative hour window vs ICU intime
    # If stop is missing, assume single-hour event at start
    j["st"] = j["starttime"]
    j["et"] = j["stoptime"].fillna(j["starttime"])

    # Clip to ICU first 24h window
    j["rel_start_hr"] = (j["st"] - j["intime"]).dt.total_seconds() / 3600.0
    j["rel_end_hr"]   = (j["et"] - j["intime"]).dt.total_seconds() / 3600.0

    # Overlap with [0, 24)
    j["hs"] = j["rel_start_hr"].clip(lower=0, upper=HOURS-1e-6)
    j["he"] = j["rel_end_hr"].clip(lower=0, upper=HOURS-1e-6)

    # Keep only rows that overlap the first 24h at all
    j = j[(j["he"] >= 0) & (j["hs"] < HOURS)].copy()

    # Convert to integer hour bins
    j["hour_start"] = j["hs"].apply(clamp_hour)
    j["hour_end"]   = j["he"].apply(clamp_hour)

    # Drop rows that failed hour calculation
    j = j[(j["hour_start"].notna()) & (j["hour_end"].notna())].copy()
    j["hour_start"] = j["hour_start"].astype(int)
    j["hour_end"]   = j["hour_end"].astype(int)

    out = j[["stay_id","drug_text","hour_start","hour_end"]].dropna().drop_duplicates()
    out.to_csv(OUTCSV, index=False)
    print(f"[extract_meds] Wrote {OUTCSV} with {len(out)} rows")

if __name__ == "__main__":
    main()
