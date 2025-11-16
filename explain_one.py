# explain_one.py — narrative single-case explainer aligned to CV training
import argparse, json, numpy as np, pandas as pd, joblib, lightgbm as lgb
from pathlib import Path

# ======== Model artifacts from CV training ========
MODEL_DIR   = Path("models_lgbm_cv")
FOLD        = 1
FOLD_MODEL  = MODEL_DIR / f"lgbm_fold{FOLD}.txt"
FOLD_CAL    = MODEL_DIR / f"calibrator_fold{FOLD}.pkl"
FN_ALL      = MODEL_DIR / "feature_names_all.json"

# ======== Base features (must mirror trainer) ========
X_TAB = Path("X_tab.npy");            FN_TAB  = Path("feature_names_tab.json")
X_SCO = Path("X_scores.npy");         FN_SCO  = Path("feature_names_scores.json")
X_PATH= Path("X_pathway.npy");        FN_PATH = Path("feature_names_pathway.json")
X_MF  = Path("med_flags_24h.npy");    FN_MF   = Path("med_flag_names.json")    # (N,24,K) or (N,K)

# ======== Hourly tensor for narrative summary (optional) ========
X_SEQ     = Path("X_data.npy")          # (N, 24, F)
FEAT_SEQ  = Path("feature_names.json")  # names for X_data channels
SPLIT     = Path("split_index.csv")
Y_LAB     = Path("y_labels.npy")

# ---------------- helpers ----------------
def _load_json(path: Path): return json.loads(path.read_text())
def _prefix(names, pfx):    return [f"{pfx}{n}" for n in names]

def _normalize_block(Xb: np.ndarray, block_label: str) -> np.ndarray:
    if Xb.ndim == 1:  return Xb.reshape(-1, 1)
    if Xb.ndim == 2:  return Xb
    if Xb.ndim == 3:
        if block_label == "med":  return np.nanmax(Xb, axis=1)
        else:                     return np.nanmean(Xb, axis=1)
    raise ValueError(f"Unexpected ndim for {block_label}: {Xb.ndim}")

def _assert_unique(names):
    seen, dups = set(), []
    for n in names:
        if n in seen: dups.append(n)
        seen.add(n)
    if dups:
        raise RuntimeError(f"Duplicate feature names after prefixing: {sorted(set(dups))[:10]}")

def _build_inference_matrix():
    assert X_TAB.exists(), "X_tab.npy missing. Run your feature builders first."
    X   = np.load(X_TAB)
    nam = _prefix(_load_json(FN_TAB), "tab__")

    if X_SCO.exists() and FN_SCO.exists():
        Xs  = _normalize_block(np.load(X_SCO), "scores")
        nam += _prefix(_load_json(FN_SCO), "score__")
        X   = np.concatenate([X, Xs], axis=1)

    if X_PATH.exists() and FN_PATH.exists():
        Xp  = _normalize_block(np.load(X_PATH), "pathway")
        nam += _prefix(_load_json(FN_PATH), "path__")
        X   = np.concatenate([X, Xp], axis=1)

    if X_MF.exists() and FN_MF.exists():
        Xm  = _normalize_block(np.load(X_MF), "med")
        nam += _prefix(_load_json(FN_MF), "med__")
        X   = np.concatenate([X, Xm], axis=1)

    if X.shape[1] != len(nam):
        raise RuntimeError(f"Columns ({X.shape[1]}) != names ({len(nam)}) when building inference matrix.")
    _assert_unique(nam)
    return X, nam

def _align_to_model(X, names, names_expected):
    if len(names) != len(names_expected):
        missing = [n for n in names_expected if n not in set(names)]
        extra   = [n for n in names if n not in set(names_expected)]
        raise RuntimeError(
            f"Feature mismatch: X has {len(names)} cols but model expects {len(names_expected)}.\n"
            f"Missing (first 10): {missing[:10]}\nExtra (first 10): {extra[:10]}"
        )
    pos = {n:i for i,n in enumerate(names)}
    order = [pos[n] for n in names_expected]
    return X[:, order], names_expected

def _pick_test_index():
    if SPLIT.exists():
        df = pd.read_csv(SPLIT)
        cand = df.index[(df["split"].astype(str).str.lower()=="test")].values
        if len(cand): return int(cand[0])
    return 0

def _top_variance_hours(x24: np.ndarray, k=3):
    mu = np.nanmean(x24, axis=0)
    var_by_hour = np.nanmean((x24 - mu)**2, axis=1)
    idx = np.argsort(var_by_hour)[::-1][:k]
    return idx.tolist(), var_by_hour[idx].round(4).tolist()

# Aliases for hourly narrative only
ALIASES = {
    "map":        ["map", "mbp", "mean_bp", "mean_arterial", "mean arterial", "mean blood pressure"],
    "wbc":        ["wbc", "white blood cell", "white blood cells"],
    "hr":         ["hr", "heart_rate", "heart rate"],
    "rr":         ["rr", "resp_rate", "respiratory_rate", "respiratory rate"],
    "spo2":       ["spo2", "sao2", "o2sat", "oxygen saturation"],
    "temp":       ["temp", "temperature"],
    "lactate":    ["lactate"],
    "creatinine": ["creatinine", "cr"],
}

def _resolve_hits(ch_names, aliases=ALIASES):
    hits = {}
    low = [n.lower() for n in ch_names]
    for key, alist in aliases.items():
        for a in alist:
            for j, nlow in enumerate(low):
                if a in nlow:
                    hits[key] = j; break
            if key in hits: break
    return hits

def _val_safe(x24, h, j):
    if j is None: return None
    v = x24[h, j]
    return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

def _summarize_channels_at_hours(x24, ch_names, hours):
    rows, hits = [], _resolve_hits(ch_names)
    for h in hours:
        parts = []
        v_map = _val_safe(x24, h, hits.get("map"))
        if v_map is not None: parts.append(f"Low MAP (map={v_map:.2f})" if v_map < 70 else f"MAP {v_map:.2f}")
        v_wbc = _val_safe(x24, h, hits.get("wbc"))
        if v_wbc is not None: parts.append(f"Leukocytosis (wbc={v_wbc:.2f})" if v_wbc >= 12 else f"WBC {v_wbc:.2f}")
        v_hr  = _val_safe(x24, h, hits.get("hr"))
        if v_hr  is not None: parts.append(f"Tachycardia (hr={v_hr:.2f})" if v_hr >= 100 else f"HR {v_hr:.2f}")
        v_sp  = _val_safe(x24, h, hits.get("spo2"))
        if v_sp  is not None: parts.append(f"Low SpO2 (spo2={v_sp:.2f})" if v_sp < 92 else f"SpO2 {v_sp:.2f}")
        v_rr  = _val_safe(x24, h, hits.get("rr"))
        if v_rr  is not None: parts.append(f"Tachypnea (rr={v_rr:.2f})" if v_rr >= 24 else f"RR {v_rr:.2f}")
        v_tmp = _val_safe(x24, h, hits.get("temp"))
        if v_tmp is not None: parts.append(f"Fever (temp={v_tmp:.2f})" if v_tmp >= 38 else f"Temp {v_tmp:.2f}")
        v_lac = _val_safe(x24, h, hits.get("lactate"))
        if v_lac is not None: parts.append(f"High Lactate (lactate={v_lac:.2f})" if v_lac >= 2 else f"Lactate {v_lac:.2f}")
        v_cr  = _val_safe(x24, h, hits.get("creatinine"))
        if v_cr  is not None: parts.append(f"High Cr (cr={v_cr:.2f})" if v_cr >= 1.5 else f"Cr {v_cr:.2f}")
        rows.append((h, "; ".join(parts) if parts else "—"))
    return rows

def _summarize_meds(med_flags_24h, med_names, hours):
    out = []
    for h in hours:
        tags = []
        for j, nm in enumerate(med_names):
            if med_flags_24h[h, j] == 1:
                tags.append(nm.replace("_"," "))
        out.append((h, ", ".join(tags) if tags else "none"))
    return out

# -------- concept resolver for probes (matches reasoner) --------
def _resolve_concept_to_model_name(names, concept: str):
    c = concept.lower()
    low = [n.lower() for n in names]
    for p in [f"score__{c}", f"tab__{c}"]:
        for n, nl in zip(names, low):
            if nl == p:
                return n
    cand = [n for n, nl in zip(names, low) if nl.endswith(c)]
    return cand[0] if cand else None

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=None)
    ap.add_argument("--probe", action="append", default=[], help="name:v1,v2 or name:low:high:step (name can be exact or concept)")
    ap.add_argument("--probe-out", type=str, default=None)
    ap.add_argument("--print-prec", type=int, default=3)
    args = ap.parse_args()

    assert FOLD_MODEL.exists(), "Fold model missing — run train_lgbm_tabular_cv.py"
    assert FOLD_CAL.exists(),   "Fold calibrator missing."
    booster = lgb.Booster(model_file=str(FOLD_MODEL))
    iso     = joblib.load(FOLD_CAL)
    names_expected = _load_json(FN_ALL)

    X, names = _build_inference_matrix()
    X, names = _align_to_model(X, names, names_expected)

    case_index = args.idx if args.idx is not None else _pick_test_index()
    y_true = int(np.load(Y_LAB)[case_index]) if Y_LAB.exists() else None

    x = X[case_index]
    p = float(iso.transform(booster.predict(x.reshape(1, -1)))[0])

    print("\n--- Patient ICU Stay Analysis ---")
    if y_true is not None:
        print(f"True label: {y_true} | Mortality Prediction: {p:.{args.print_prec}f}")
    else:
        print(f"Mortality Prediction: {p:.{args.print_prec}f}")

    if X_SEQ.exists() and FEAT_SEQ.exists():
        x24 = np.load(X_SEQ)[case_index]
        ch_names = _load_json(FEAT_SEQ)
        hours, _ = _top_variance_hours(x24, k=3)
        print("Top attention hours:", ", ".join(str(h) for h in hours))
        print("Explanation:")
        for h, desc in _summarize_channels_at_hours(x24, ch_names, hours):
            print(f" - Hour {h}: {desc}")

        if X_MF.exists() and FN_MF.exists():
            H = np.load(X_MF)
            meds_24 = H[case_index] if H.ndim == 3 else np.repeat(H[case_index][None, :], 24, axis=0)
            med_names = _load_json(FN_MF)
            print("\nMedication flags at top attention hours:")
            for h, tag in _summarize_meds(meds_24, med_names, hours):
                print(f" - Hour {h}: {tag}")

    # ---------------- Counterfactual probes ----------------
    def _parse_probe(ptext: str):
        parts = ptext.split(":")
        if len(parts) == 2 and "," in parts[1]:
            name, vals = parts
            vals = [float(v) for v in vals.split(",")]
            return name, vals
        elif len(parts) == 4:
            name, lo, hi, step = parts
            import numpy as np
            vals = list(np.arange(float(lo), float(hi) + 1e-9, float(step)))
            return name, vals
        else:
            raise ValueError(f"Bad --probe format: {ptext}")

    probe_rows = []
    if args.probe:
        base = p
        for pexpr in args.probe:
            try:
                name, vals = _parse_probe(pexpr)
            except Exception as e:
                print(f"[warn] skipping bad --probe '{pexpr}': {e}"); continue

            resolved = name if name in names else _resolve_concept_to_model_name(names, name)
            if not resolved:
                print(f"[warn] probe feature '{name}' not found (even by concept); skipping."); continue

            j = {n:i for i,n in enumerate(names)}[resolved]
            for v in vals:
                x_cf = x.copy(); x_cf[j] = float(v)
                pnew = float(iso.transform(booster.predict(x_cf.reshape(1, -1)))[0])
                print(f" - Probe {resolved}={v:.3f}: risk {pnew:.{args.print_prec}f} (Δ {pnew - base:+.{args.print_prec}f})")
                probe_rows.append({"feature": resolved, "value": float(v), "risk": float(pnew), "delta": float(pnew - base)})
        if args.probe_out and probe_rows:
            pd.DataFrame(probe_rows).to_csv(args.probe_out, index=False)
            print(f"[ok] Probes saved → {args.probe_out}")
    else:
        # keep the small “toy” lines for parity when no probes are passed
        print("\nCounterfactual probes :")
        print(" - MAP +5.0 → risk (toy)")
        print(" - LACTATE -1.0 → risk (toy)")

if __name__ == "__main__":
    main()
