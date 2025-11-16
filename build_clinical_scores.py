# build_clinical_scores.py
"""
Builds clinical score-like tabular features aligned EXACTLY to cohort_index.csv.
- Reads cohort_index.csv to define the cohort and order
- Fetches first-24h hourly vitals/labs from DB
- Derives compact “scores” (shock index, pulse pressure, simple summaries)
- Saves X_scores.npy and feature_names_scores.json

You can extend/replace `compute_scores()` with your existing formulas;
the critical alignment step is the final reindex using cohort['stay_id'].
"""

import os, getpass, json
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text

# ---------- DB ----------
DB_USER = os.environ.get("PGUSER") or getpass.getuser()
engine  = create_engine(f"postgresql+psycopg2://{DB_USER}@localhost:5432/mimiciv", future=True)

# ---------- IO ----------
COHORT_CSV = Path("cohort_index.csv")     # created by build_day1_dataset.py
OUT_X      = Path("X_scores.npy")
OUT_FN     = Path("feature_names_scores.json")

HOURS      = 24

# Core signals we’ll use for “scores”
VITAL_COLS = ["heart_rate", "sbp", "dbp", "mbp", "resp_rate", "temperature", "spo2", "glucose"]
LAB_KEYS   = ["lactate", "creatinine", "bun", "wbc"]


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
            v.spo2,
            v.glucose
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
          AVG(spo2)        AS spo2,
          AVG(glucose)     AS glucose
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


def get_lab_itemids() -> dict:
    with engine.connect() as conn:
        dlab = pd.read_sql(text("SELECT itemid, label FROM mimiciv_hosp.d_labitems"), conn)
    dlab["lbl"] = dlab["label"].fillna("").str.lower()

    patterns = {
        "lactate":    ["lactate"],
        "creatinine": ["creatinine"],
        "bun":        ["urea nitrogen", "bun"],
        "wbc":        ["wbc", "white blood cell", "white blood cells"],
    }
    mapping = {}
    for out_name, pats in patterns.items():
        pats = [p.lower() for p in pats]
        sub = dlab[dlab["lbl"].apply(lambda s: any(p in s for p in pats))]
        mapping[out_name] = sorted(sub["itemid"].unique().tolist())
    return mapping


def fetch_hourly_labs(cohort: pd.DataFrame, lab_itemids: dict) -> pd.DataFrame:
    all_rows = []
    values_list = ",".join([f"(:sid{k}, :t{k})" for k in range(len(cohort))])

    with engine.connect() as conn:
        for out_name, ids in lab_itemids.items():
            if not ids:
                continue
            sql = text(f"""
                WITH l0 AS (
                  SELECT i.stay_id, l.charttime, l.valuenum
                  FROM mimiciv_hosp.labevents l
                  JOIN mimiciv_icu.icustays i
                    USING (subject_id, hadm_id)
                  WHERE l.itemid = ANY(:ids)
                ),
                j AS (
                  SELECT
                    c.stay_id,
                    FLOOR(EXTRACT(EPOCH FROM (l0.charttime - c.intime))/3600.0) AS relhr,
                    l0.valuenum
                  FROM l0
                  JOIN (VALUES {values_list}) AS c(stay_id,intime)
                    USING (stay_id)
                )
                SELECT stay_id, relhr, AVG(valuenum) AS {out_name}
                FROM j
                WHERE relhr BETWEEN 0 AND {HOURS-1}
                GROUP BY stay_id, relhr
                ORDER BY stay_id, relhr;
            """)
            params = {"ids": ids}
            for k, (sid, intime) in enumerate(zip(cohort["stay_id"], cohort["intime"])):
                params[f"sid{k}"] = int(sid)
                params[f"t{k}"]   = pd.Timestamp(intime)
            all_rows.append(pd.read_sql(sql, conn, params=params))

    if not all_rows:
        return pd.DataFrame(columns=["stay_id", "relhr"])

    out = all_rows[0]
    for add in all_rows[1:]:
        out = out.merge(add, on=["stay_id", "relhr"], how="outer")
    return out.sort_values(["stay_id", "relhr"])


def summarize_first24h(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    """
    df: long per (stay_id, relhr) with `keys` columns
    returns wide per stay with summary stats per key
    """
    df = df.sort_values(["stay_id", "relhr"]).copy()

    def _last(s: pd.Series):
        # assumes df is sorted by relhr
        return s.iloc[-1] if len(s) else np.nan

    grouped = df.groupby("stay_id", as_index=False)

    out = None
    for k in keys:
        # aggregate this single column, then rename
        gk = (
            grouped[k]
            .agg(min_val="min", max_val="max", mean_val="mean", last_val=_last, std_val="std")
            .rename(columns={
                "min_val":  f"{k}__min",
                "max_val":  f"{k}__max",
                "mean_val": f"{k}__mean",
                "last_val": f"{k}__last",
                "std_val":  f"{k}__std",
            })
        )
        # first iteration seeds the output
        if out is None:
            out = gk
        else:
            out = out.merge(gk, on="stay_id", how="outer")

    return out


def compute_scores(vitsum: pd.DataFrame) -> pd.DataFrame:
    """
    Create compact “score” features from vitals/labs summaries.
    This is intentionally lightweight so it runs everywhere.
    """
    df = vitsum.copy()

    # Shock index (mean HR / mean SBP)
    if {"heart_rate__mean","sbp__mean"}.issubset(df.columns):
        df["shock_index_mean"] = df["heart_rate__mean"] / (df["sbp__mean"].replace(0, np.nan))
    else:
        df["shock_index_mean"] = np.nan

    # Pulse pressure (mean SBP - mean DBP)
    if {"sbp__mean","dbp__mean"}.issubset(df.columns):
        df["pulse_pressure_mean"] = df["sbp__mean"] - df["dbp__mean"]
    else:
        df["pulse_pressure_mean"] = np.nan

    # Hypoxemia burden approx: 100 - mean SpO2
    if "spo2__mean" in df.columns:
        df["hypoxemia_burden"] = 100.0 - df["spo2__mean"]
    else:
        df["hypoxemia_burden"] = np.nan

    # Hyperglycemia indicator: (mean glucose)
    if "glucose__mean" in df.columns:
        df["glucose_mean"] = df["glucose__mean"]
    else:
        df["glucose_mean"] = np.nan

    # Lactate mean (if available)
    if "lactate__mean" in df.columns:
        df["lactate_mean"] = df["lactate__mean"]
    else:
        df["lactate_mean"] = np.nan

    # Clean infs/nans
    df = df.replace([np.inf, -np.inf], np.nan).fillna(df.median(numeric_only=True))

    keep_cols = [c for c in df.columns if c != "stay_id"]
    return df[["stay_id"] + keep_cols]


def main():
    # 1) Load the cohort to define patient set + order
    cohort = pd.read_csv(COHORT_CSV)
    if "stay_id" not in cohort.columns or "intime" not in cohort.columns:
        raise RuntimeError("cohort_index.csv must contain stay_id and intime columns")

    # 2) Pull first-24h vitals & labs
    vit = fetch_hourly_vitals(cohort)
    lab_ids = get_lab_itemids()
    labs = fetch_hourly_labs(cohort, lab_ids)

    # 3) Join and summarize per stay
    long = (
        pd.merge(vit, labs, on=["stay_id","relhr"], how="left")
          .sort_values(["stay_id","relhr"])
    )
    have_cols = [c for c in (VITAL_COLS + LAB_KEYS) if c in long.columns]
    for c in have_cols:
        long[c] = pd.to_numeric(long[c], errors="coerce")

    vitsum = summarize_first24h(long[["stay_id","relhr"] + have_cols], have_cols)

    # 4) Build compact score features
    scores = compute_scores(vitsum)  # has 'stay_id'

    # 5) **CRITICAL**: align to cohort order & fill
    scores = scores.set_index("stay_id").reindex(cohort["stay_id"]).reset_index()
    scores = scores.fillna(scores.median(numeric_only=True)).fillna(0)

    # 6) Save
    names = [c for c in scores.columns if c != "stay_id"]
    X = scores[names].to_numpy(np.float32)
    np.save(OUT_X, X)
    OUT_FN.write_text(json.dumps(names, indent=2))

    print(f"[scores] X_scores.npy shape={X.shape}")
    print(f"[scores] feature_names_scores.json count={len(names)}")
    print("[scores] Alignment OK with cohort_index.csv")

if __name__ == "__main__":
    main()
