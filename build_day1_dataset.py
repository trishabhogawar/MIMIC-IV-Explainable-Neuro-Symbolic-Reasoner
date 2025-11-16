# build_day1_dataset.py
import os, getpass, json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, List

# ---------- DB ----------
DB_USER = os.environ.get("PGUSER") or getpass.getuser()
engine  = create_engine(f"postgresql+psycopg2://{DB_USER}@localhost:5432/mimiciv", future=True)

# ---------- CONFIG ----------
N_PATIENTS   = 65366
HOURS        = 24

# Expanded vitals. We will SELECT COALESCE for sbp/dbp/mbp so we only keep one each.
# (heart_rate, resp_rate, temperature, spo2, glucose retained as-is)
VITAL_CANON = ["heart_rate", "sbp", "dbp", "mbp", "resp_rate", "temperature", "spo2", "glucose"]

LAB_LABELS   = {
    "lactate":    ["lactate"],
    "creatinine": ["creatinine"],
    "bun":        ["urea nitrogen", "bun"],
    "wbc":        ["wbc", "white blood cell", "white blood cells"],
}

# outputs (time-series)
OUT_X      = "X_data.npy"            # (N, 24, F_ts)
OUT_Y      = "y_labels.npy"
OUT_META   = "cohort_index.csv"
OUT_SCALER = "scaler.pkl"
OUT_FEATS  = "feature_names.json"    # time-series feature names (order in X_data)

FLAGS_CSV  = "med_flags.csv"         # optional causal med flags (aligned by stay_id)


def cohort_first_icu_adults(n=N_PATIENTS) -> pd.DataFrame:
    sql = text("""
        WITH first_icu AS (
          SELECT subject_id, hadm_id, stay_id, intime, outtime,
                 ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY intime) AS rn
          FROM mimiciv_icu.icustays
        )
        SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime,
               a.hospital_expire_flag::int AS mortality
        FROM first_icu i
        JOIN mimiciv_hosp.patients  p USING (subject_id)
        JOIN mimiciv_hosp.admissions a USING (hadm_id)
        WHERE i.rn = 1 AND p.anchor_age >= 18
        ORDER BY i.intime
        LIMIT :n
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn, params={"n": int(n)})


def get_lab_itemids() -> Dict[str, List[int]]:
    with engine.connect() as conn:
        dlab = pd.read_sql(text("SELECT itemid, label FROM mimiciv_hosp.d_labitems"), conn)
    dlab["lbl"] = dlab["label"].fillna("").str.lower()

    mapping = {}
    for out_name, patterns in LAB_LABELS.items():
        pats = [p.lower() for p in patterns]
        sub = dlab[dlab["lbl"].apply(lambda s: any(p in s for p in pats))]
        mapping[out_name] = sorted(sub["itemid"].unique().tolist())
    print("Lab itemids resolved:", {k: len(v) for k, v in mapping.items()})
    return mapping


def fetch_hourly_vitals(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Pulls hourly vitals for first 24h using invasive if present, else NI fallback:
      sbp := COALESCE(sbp, sbp_ni)
      dbp := COALESCE(dbp, dbp_ni)
      mbp := COALESCE(mbp, mbp_ni)
    """
    values_list = ",".join([f"(:sid{i}, :t{i})" for i in range(len(cohort))])

    # We select all columns we care about from mimiciv_derived.vitalsign
    # and compute hourly averages per stay_id, relhr ∈ [0..23]
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
          stay_id,
          relhr,
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
                  JOIN mimiciv_icu.icustays i USING (subject_id, hadm_id)
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


def make_grid(cohort: pd.DataFrame) -> pd.DataFrame:
    return pd.MultiIndex.from_product(
        [cohort["stay_id"].tolist(), range(HOURS)],
        names=["stay_id", "relhr"]
    ).to_frame(index=False)


def forward_then_backfill(df: pd.DataFrame, by: str = "stay_id") -> pd.DataFrame:
    df = df.sort_values([by, "relhr"]).copy()
    value_cols = [c for c in df.columns if c not in ("stay_id", "relhr")]
    for c in value_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in value_cols:
        df[c] = df.groupby(by, group_keys=False)[c].transform(lambda s: s.ffill().bfill())
        med = df[c].median()
        df[c] = df[c].fillna(med)
    return df


def maybe_load_flags(cohort: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(FLAGS_CSV):
        print(f"[Info] {FLAGS_CSV} not found; skipping causal flags.")
        return pd.DataFrame()

    flags = pd.read_csv(FLAGS_CSV)
    required = {"stay_id", "cause_vasopressor", "cause_beta_blocker", "cause_sedation"}
    if not required.issubset(set(flags.columns)):
        print(f"[Warn] {FLAGS_CSV} missing required columns {required}; skipping flags.")
        return pd.DataFrame()

    flags = flags.set_index("stay_id").reindex(cohort["stay_id"]).fillna(0).reset_index(drop=True)
    for c in ["cause_vasopressor","cause_beta_blocker","cause_sedation"]:
        flags[c] = pd.to_numeric(flags[c], errors="coerce").fillna(0).clip(0,1).astype(np.float32)
    return flags[["cause_vasopressor","cause_beta_blocker","cause_sedation"]]


def main():
    print("Building cohort...")
    cohort = cohort_first_icu_adults()
    cohort = cohort[['subject_id','hadm_id','stay_id','intime','outtime','mortality']].reset_index(drop=True)

    print("Resolving lab itemids...")
    lab_ids = get_lab_itemids()

    print("Fetching hourly vitals (with NI fallback)...")
    vit = fetch_hourly_vitals(cohort)

    print("Fetching hourly labs...")
    labs = fetch_hourly_labs(cohort, lab_ids)

    print("Aligning to 24h grid and filling values...")
    grid = make_grid(cohort)
    data = grid.merge(vit,  on=["stay_id","relhr"], how="left")
    data = data.merge(labs, on=["stay_id","relhr"], how="left")
    data = forward_then_backfill(data, by="stay_id")

    # feature order: all vitals, then 4 labs (if present), then causal flags
    feature_cols = [c for c in VITAL_CANON if c in data.columns]
    for c in ["lactate","creatinine","bun","wbc"]:
        if c in data.columns:
            feature_cols.append(c)
    if not feature_cols:
        raise RuntimeError("No features resolved!")

    # tensorize (N, 24, F)
    data = data.sort_values(["stay_id","relhr"])
    stay_to_idx = {sid:i for i,sid in enumerate(cohort["stay_id"].tolist())}
    X = np.zeros((len(cohort), HOURS, len(feature_cols)), dtype=np.float32)
    for _, row in data.iterrows():
        i = stay_to_idx[row["stay_id"]]; h = int(row["relhr"])
        X[i, h, :] = row[feature_cols].to_numpy(dtype=float)

    y = cohort["mortality"].to_numpy(dtype=np.int64)

    # Optional flags broadcast across hours
    flags = maybe_load_flags(cohort)
    if not flags.empty:
        flag_arr = np.repeat(flags.to_numpy(dtype=np.float32)[:,None,:], repeats=X.shape[1], axis=1)
        X = np.concatenate([X, flag_arr], axis=-1)
        feature_cols.extend(["cause_vasopressor","cause_beta_blocker","cause_sedation"])
        print(f"Appended causal flags. New feature count: {len(feature_cols)}")

    # global scaling (you also fit per-fold scalers downstream; this just stabilizes TS)
    scaler = StandardScaler()
    X2 = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    np.save(OUT_X, X2)
    np.save(OUT_Y, y)
    cohort[["subject_id","hadm_id","stay_id","intime","mortality"]].to_csv(OUT_META, index=False)
    joblib.dump(scaler, OUT_SCALER)
    with open(OUT_FEATS, "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"Saved: {OUT_X} shape={X2.shape}")
    print(f"Saved: {OUT_Y}")
    print(f"Saved: {OUT_META}, {OUT_SCALER}, {OUT_FEATS}")

if __name__ == "__main__":
    main()
