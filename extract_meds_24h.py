# extract_meds_24h.py
"""
Create meds_24h.csv aligned to cohort_index.csv
Outputs columns: stay_id, drug_text, hour_start, hour_end
"""

import os, getpass
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine, text

COHORT = Path("cohort_index.csv")
OUTCSV = Path("meds_24h.csv")
HOURS  = 24

DB_USER = os.environ.get("PGUSER") or getpass.getuser()
engine  = create_engine(f"postgresql+psycopg2://{DB_USER}@localhost:5432/mimiciv", future=True)

# Try these in order; we’ll pick the first one that exists in prescriptions
DRUG_NAME_CANDIDATES = [
    "drug_name_generic", "drug", "drug_name_poe", "formulary_drug_cd", "product", "medication"
]

def clamp_hour(x):
    if pd.isna(x): return None
    try: x = int(np.floor(x))
    except Exception: return None
    return max(0, min(23, x))

def pick_available_drug_col(conn) -> str:
    q = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='mimiciv_hosp' AND table_name='prescriptions'
    """)
    cols = pd.read_sql(q, conn)["column_name"].str.lower().tolist()
    for c in DRUG_NAME_CANDIDATES:
        if c.lower() in cols:
            return c
    # fallback to any column if none matched (unlikely)
    if cols: return cols[0]
    raise RuntimeError("No columns found in mimiciv_hosp.prescriptions")

def main():
    if not COHORT.exists():
        raise FileNotFoundError("cohort_index.csv not found. Run build_day1_dataset.py first.")
    cohort = pd.read_csv(COHORT)
    req = {"stay_id","hadm_id","intime"}
    if not req.issubset(cohort.columns):
        raise RuntimeError(f"cohort_index.csv must contain {req}")

    cohort["intime"] = pd.to_datetime(cohort["intime"], errors="coerce", utc=True)
    stays = cohort[["stay_id","hadm_id","intime"]].drop_duplicates()
    print(f"[extract_meds] Cohort rows: {len(stays)}")

    hadm_ids = stays["hadm_id"].astype(int).unique().tolist()
    if not hadm_ids:
        OUTCSV.write_text("")
        print(f"[extract_meds] No admissions; wrote empty {OUTCSV}")
        return

    CHUNK = 5000
    rows = []
    with engine.connect() as conn:
        drug_col = pick_available_drug_col(conn)
        print(f"[extract_meds] Using prescriptions.{drug_col} as drug_text")

        sql_tpl = f"""
            SELECT
                p.hadm_id,
                p.starttime,
                p.stoptime,
                TRIM(p.{drug_col}) AS drug_text
            FROM mimiciv_hosp.prescriptions p
            WHERE p.hadm_id = ANY(:hadms)
        """
        for i in range(0, len(hadm_ids), CHUNK):
            chunk = hadm_ids[i:i+CHUNK]
            df = pd.read_sql(text(sql_tpl), conn, params={"hadms": chunk})
            rows.append(df)

    presc = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["hadm_id","starttime","stoptime","drug_text"]
    )

    # Clean
    presc["starttime"] = pd.to_datetime(presc["starttime"], errors="coerce", utc=True)
    presc["stoptime"]  = pd.to_datetime(presc["stoptime"],  errors="coerce", utc=True)
    presc["drug_text"] = presc["drug_text"].astype(str).str.strip().str.lower()
    presc = presc[~presc["drug_text"].isin(["", "none", "nan"])].copy()

    # Join to ICU stays
    j = presc.merge(stays, on="hadm_id", how="inner")

    # Relative hours vs ICU intime
    j["st"] = j["starttime"]
    j["et"] = j["stoptime"].fillna(j["starttime"])
    j["rel_start_hr"] = (j["st"] - j["intime"]).dt.total_seconds() / 3600.0
    j["rel_end_hr"]   = (j["et"] - j["intime"]).dt.total_seconds() / 3600.0

    # Overlap with [0,24)
    j["hs"] = j["rel_start_hr"].clip(lower=0, upper=HOURS-1e-6)
    j["he"] = j["rel_end_hr"].clip(lower=0, upper=HOURS-1e-6)
    j = j[(j["he"] >= 0) & (j["hs"] < HOURS)].copy()

    # Integer bins
    j["hour_start"] = j["hs"].apply(clamp_hour)
    j["hour_end"]   = j["he"].apply(clamp_hour)
    j = j[(j["hour_start"].notna()) & (j["hour_end"].notna())].copy()
    j["hour_start"] = j["hour_start"].astype(int)
    j["hour_end"]   = j["hour_end"].astype(int)

    out = j[["stay_id","drug_text","hour_start","hour_end"]].dropna().drop_duplicates()
    out.to_csv(OUTCSV, index=False)
    print(f"[extract_meds] Wrote {OUTCSV} with {len(out)} rows")

if __name__ == "__main__":
    main()
