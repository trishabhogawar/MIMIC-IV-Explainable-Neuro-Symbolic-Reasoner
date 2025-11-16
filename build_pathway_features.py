# build_pathway_features.py
"""
Creates simple event/pathway features from first-24h vitals, aligned to cohort_index.csv.
Examples:
  - Any hypotension (SBP<90) event; duration (hours), count of hours
  - Any tachycardia (HR>120); hypoxemia (SpO2<90); fever (Temp>=38.3); bradycardia (HR<50)
You can extend the rules freely; alignment is guaranteed by the final reindex.
"""

import os, getpass, json
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

# ---------- DB ----------
DB_USER = os.environ.get("PGUSER") or getpass.getuser()
engine  = create_engine(f"postgresql+psycopg2://{DB_USER}@localhost:5432/mimiciv", future=True)

COHORT_CSV = Path("cohort_index.csv")
OUT_X      = Path("X_pathway.npy")
OUT_FN     = Path("feature_names_pathway.json")
HOURS      = 24

def fetch_hourly_vitals(cohort: pd.DataFrame) -> pd.DataFrame:
    values_list = ",".join([f"(:sid{i}, :t{i})" for i in range(len(cohort))])
    sql = text(f"""
        WITH base AS (
          SELECT
            v.stay_id,
            FLOOR(EXTRACT(EPOCH FROM (v.charttime - c.intime))/3600.0) AS relhr,
            v.heart_rate,
            COALESCE(v.sbp, v.sbp_ni) AS sbp,
            COALESCE(v.dbp, v.dbp_ni) AS dbp,
            COALESCE(v.mbp, v.mbp_ni) AS mbp,
            v.resp_rate,
            v.temperature,
            v.spo2
          FROM mimiciv_derived.vitalsign v
          JOIN (VALUES {values_list}) AS c(stay_id,intime)
            ON c.stay_id = v.stay_id
          WHERE v.charttime >= c.intime
            AND v.charttime <  c.intime + INTERVAL '{HOURS} hour'
        )
        SELECT
          stay_id, relhr,
          AVG(heart_rate)  AS heart_rate,
          AVG(sbp)         AS sbp,
          AVG(dbp)         AS dbp,
          AVG(mbp)         AS mbp,
          AVG(resp_rate)   AS resp_rate,
          AVG(temperature) AS temperature,
          AVG(spo2)        AS spo2
        FROM base
        WHERE relhr BETWEEN 0 AND {HOURS-1}
        GROUP BY stay_id, relhr
        ORDER BY stay_id, relhr;
    """)
    params = {}
    for i, (sid, intime) in enumerate(zip(cohort["stay_id"], cohort["intime"])):
        params[f"sid{i}"] = int(sid)
        params[f"t{i}"]   = pd.Timestamp(intime)

    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params=params)


def make_pathway_features(v: pd.DataFrame) -> pd.DataFrame:
    """
    v: long per (stay_id, relhr) with columns heart_rate, sbp, dbp, mbp, resp_rate, temperature, spo2
    Returns wide per-stay pathway feature set.
    """
    v = v.sort_values(["stay_id","relhr"]).copy()
    for c in ["heart_rate","sbp","dbp","mbp","resp_rate","temperature","spo2"]:
        if c in v.columns:
            v[c] = pd.to_numeric(v[c], errors="coerce")

    # Define boolean events per hour
    v["hypotension_hr"]   = (v["sbp"] < 90).astype(float)
    v["tachycardia_hr"]   = (v["heart_rate"] > 120).astype(float)
    v["bradycardia_hr"]   = (v["heart_rate"] < 50).astype(float)
    v["hypoxemia_hr"]     = (v["spo2"] < 90).astype(float)
    v["fever_hr"]         = (v["temperature"] >= 38.3).astype(float)
    v["tachypnea_hr"]     = (v["resp_rate"] > 30).astype(float)

    # Aggregate per stay
    grp = v.groupby("stay_id")
    def dur(x):   return float(np.nansum(x))                  # hours with event
    def any_(x):  return float(np.nanmax(x)) if len(x) else 0 # any event flag
    def first_onset(x):
        idx = np.where(np.nan_to_num(x, nan=0.0) > 0.5)[0]
        return float(idx.min()) if idx.size>0 else float(HOURS)

    feats = grp.agg(
        hypotension_any=("hypotension_hr", any_),
        hypotension_hours=("hypotension_hr", dur),
        hypotension_first_hr=("hypotension_hr", first_onset),

        tachycardia_any=("tachycardia_hr", any_),
        tachycardia_hours=("tachycardia_hr", dur),
        tachycardia_first_hr=("tachycardia_hr", first_onset),

        bradycardia_any=("bradycardia_hr", any_),
        bradycardia_hours=("bradycardia_hr", dur),
        bradycardia_first_hr=("bradycardia_hr", first_onset),

        hypoxemia_any=("hypoxemia_hr", any_),
        hypoxemia_hours=("hypoxemia_hr", dur),
        hypoxemia_first_hr=("hypoxemia_hr", first_onset),

        fever_any=("fever_hr", any_),
        fever_hours=("fever_hr", dur),
        fever_first_hr=("fever_hr", first_onset),

        tachypnea_any=("tachypnea_hr", any_),
        tachypnea_hours=("tachypnea_hr", dur),
        tachypnea_first_hr=("tachypnea_hr", first_onset),
    ).reset_index()

    # Add simple trend feature: SBP slope across 24h (per stay)
    def sbp_slope(stay_df):
        x = stay_df.sort_values("relhr")["sbp"].to_numpy()
        t = np.arange(len(x), dtype=np.float32)
        t = (t - t.mean()) / (t.std() + 1e-8)
        cov = (x * t).sum() / max(1, len(x)-1)
        var = (t**2).sum() / max(1, len(x)-1)
        return float(cov / (var + 1e-8))
    slope_df = v.groupby("stay_id").apply(sbp_slope).rename("sbp_slope").reset_index()

    feats = feats.merge(slope_df, on="stay_id", how="left")
    return feats


def main():
    # 1) Cohort order is the source of truth
    cohort = pd.read_csv(COHORT_CSV)
    if "stay_id" not in cohort.columns or "intime" not in cohort.columns:
        raise RuntimeError("cohort_index.csv must contain stay_id and intime columns")

    # 2) Fetch first-24h hourly vitals
    vit = fetch_hourly_vitals(cohort)

    # 3) Build pathway features
    feats = make_pathway_features(vit)  # has 'stay_id'

    # 4) **CRITICAL**: align to cohort order & fill
    feats = feats.set_index("stay_id").reindex(cohort["stay_id"]).reset_index()
    feats = feats.fillna(feats.median(numeric_only=True)).fillna(0)

    # 5) Save
    names = [c for c in feats.columns if c != "stay_id"]
    X = feats[names].to_numpy(np.float32)
    np.save(OUT_X, X)
    OUT_FN.write_text(json.dumps(names, indent=2))

    print(f"[pathway] X_pathway.npy shape={X.shape}")
    print(f"[pathway] feature_names_pathway.json count={len(names)}")
    print("[pathway] Alignment OK with cohort_index.csv")

if __name__ == "__main__":
    main()
