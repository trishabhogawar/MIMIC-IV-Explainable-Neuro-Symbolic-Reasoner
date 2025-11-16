# neuro_symbolic_reasoner.py
import argparse, json, joblib, numpy as np, lightgbm as lgb
from pathlib import Path
from typing import List, Dict, Any, Tuple
import os, itertools

# Reuse your training-style builders (already in your repo)
from shap_demo import build_features, choose_model_and_expected_names, align_to_model
from causal_rules import evaluate_rules
from causal_graph import apply_do_intervention, RULE_TO_NODE, dag_paths_to_mortality

# ---------------- core math ----------------
def sigmoid(z): 
    return 1.0 / (1.0 + np.exp(-z))

def logit(p, eps=1e-6): 
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def combine_neuro_symbolic(p_ml: float, fired_rules, alpha: float = 1.0) -> float:
    """
    Combine neural probability with symbolic rule weights in logit space:
      logit(p_ns) = logit(p_ml) + alpha * sum(weight_i)
    """
    z = logit(p_ml)
    z += alpha * sum(r.weight for r in fired_rules)
    return float(sigmoid(z))

def explain_rules(fired_rules) -> List[str]:
    out = []
    for r in fired_rules:
        ev = "; ".join(r.evidence)
        out.append(f"Rule '{r.name}' fired [{r.weight:+.2f}] due to: {ev}")
    return out

# ---------------- helpers (counterfactual-only) ----------------
def _parse_kv_edits(pairs: List[str]) -> Dict[str, Any]:
    edits: Dict[str, Any] = {}
    for s in pairs:
        if "=" in s:
            k, v = s.split("=", 1)
            try: v = float(v)
            except Exception: pass
            edits[k] = v
    return edits

def _parse_range(expr: str) -> Tuple[str, np.ndarray]:
    """
    'score__mbp__mean:55:75:5' or 'mbp__mean:55:75:5'
    """
    name, lo, hi, step = expr.split(":")
    vals = np.arange(float(lo), float(hi) + 1e-9, float(step))
    return name, vals

def _med_columns(feature_names: List[str]) -> List[str]:
    return [c for c in feature_names if c.startswith("med__")]

def _apply_edits_vec(x_row: np.ndarray, names: List[str], edits: Dict[str, Any]) -> np.ndarray:
    """Return a modified copy by setting features named in 'edits'."""
    pos = {n: i for i, n in enumerate(names)}
    x = x_row.copy()
    for k, v in edits.items():
        if k in pos:
            x[pos[k]] = float(v)
    return x

def _resolve_concept_to_model_name(names: List[str], concept: str):
    """
    Allow short/semantic targets like 'mbp__mean' or 'lactate__mean'.
    Prefer score__* over tab__* if both exist; else fallback to suffix match.
    Returns the resolved model column name or None.
    """
    c = concept.lower()
    low = [n.lower() for n in names]
    for p in [f"score__{c}", f"tab__{c}"]:
        for n, nl in zip(names, low):
            if nl == p:
                return n
    cand = [n for n, nl in zip(names, low) if nl.endswith(c)]
    return cand[0] if cand else None

def _family_aliases(root: str):
    r = root.lower()
    if r in {"mbp", "map", "mean_bp"}: return ["mbp", "map", "mean_bp"]
    if r in {"spo2", "o2sat", "sao2"}: return ["spo2", "o2sat", "sao2"]
    if r in {"hr", "heart_rate"}:      return ["hr", "heart_rate"]
    if r in {"rr", "resp_rate"}:       return ["rr", "resp_rate", "respiratory"]
    if r in {"lactate"}:               return ["lactate"]
    return [r]

def _apply_family_edit(x_row, names, concept_or_name: str, value: float, also_z: bool = True):
    """
    When --edit-family is used: set any column whose name contains the concept family.
    Also nudge any '*__z' feature that matches the family (rough proxy if trees split on z).
    """
    x = x_row.copy()
    low = [n.lower() for n in names]
    root = concept_or_name.split("__", 1)[0]
    keys = _family_aliases(root)
    for i, nlow in enumerate(low):
        if any(k in nlow for k in keys):
            if concept_or_name.lower() in nlow or "__mean" in concept_or_name.lower():
                x[i] = float(value)
            if also_z and nlow.endswith("__z") and any(k in nlow for k in keys):
                x[i] = -2.0 if value < 60 else (-1.0 if value < 70 else 0.0)
    return x

# ------ NEW: multi-scenario meds parsing ------
def _parse_med_scenario(s: str) -> Tuple[str, Dict[str, float]]:
    """
    'label|med__a=0,med__b=1'  -> ('label', {'med__a':0.0,'med__b':1.0})
    'med__a=0,med__b=1'        -> ('med__a=0,med__b=1', {...})
    """
    if "|" in s:
        label, rest = s.split("|", 1)
    else:
        label, rest = s, s
    edits: Dict[str, float] = {}
    for tok in rest.split(","):
        tok = tok.strip()
        if not tok: 
            continue
        if "=" not in tok:
            raise ValueError(f"Bad med-scenario token '{tok}' (expected key=val).")
        k, v = tok.split("=", 1)
        edits[k.strip()] = float(v)
    return label.strip(), edits

def _generate_med_grid(names: List[str], med_list: List[str], values: List[float], max_scenarios: int):
    """
    Produce (label, edits_dict) for all combinations of given meds over 'values'.
    """
    # Keep only meds that exist
    valid_meds = [m for m in med_list if m in names]
    if len(valid_meds) < len(med_list):
        missing = [m for m in med_list if m not in valid_meds]
        print(f"[warn] med(s) not found and will be ignored: {missing}")
    if not valid_meds:
        return []

    combos = list(itertools.product(values, repeat=len(valid_meds)))
    if len(combos) > max_scenarios:
        print(f"[warn] grid would create {len(combos)} scenarios; capping to first {max_scenarios}.")
        combos = combos[:max_scenarios]

    scenarios = []
    for combo in combos:
        edits = {m: float(v) for m, v in zip(valid_meds, combo)}
        label = ",".join(f"{m}={int(v) if float(v).is_integer() else v}" for m, v in edits.items())
        scenarios.append((label, edits))
    return scenarios

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models_lgbm_cv")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--do", nargs="*", default=[], help="edits e.g. med__pressor=0 score__mbp__mean=70 or mbp__mean=70")
    ap.add_argument("--debug", action="store_true")

    # -------- sweeps & UX flags --------
    ap.add_argument("--sweep-meds", choices=["eachoff","eachon","alloff","allon"], default=None)
    ap.add_argument("--sweep-meds-include", type=str, default=None)
    ap.add_argument("--sweep-range", action="append", default=[], help="name:low:high:step; name can be exact or concept")
    ap.add_argument("--sweep-out", type=str, default=None)
    ap.add_argument("--edit-family", action="store_true", help="Edit related alias columns and z-variants too.")
    ap.add_argument("--print-prec", type=int, default=3)

    # -------- NEW: multiple medication variations in one run --------
    ap.add_argument("--med-scenario", action="append", default=[],
                    help="Repeatable. 'label|med__a=0,med__b=1' or 'med__a=0,med__b=1'")
    ap.add_argument("--med-grid", type=str, default=None,
                    help="Comma list of meds to generate all combinations (e.g., 'med__pressor,med__sedation').")
    ap.add_argument("--grid-values", type=str, default="0,1",
                    help="Values for combinations, comma list (default '0,1').")
    ap.add_argument("--grid-max", type=int, default=128,
                    help="Cap the number of auto-generated scenarios (default 128).")

    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    booster, names_expected = choose_model_and_expected_names(model_dir, args.fold)
    iso = joblib.load(model_dir / f"calibrator_fold{args.fold}.pkl")

    X, names = build_features()
    X, names = align_to_model(X, names, names_expected)

    i = int(args.idx)
    x  = X[i]
    p_ml = float(iso.transform(booster.predict(x.reshape(1,-1)))[0])

    from causal_rules import debug_snapshot
    snap = debug_snapshot(x, names)

    if args.debug:
        print("\n[DEBUG] resolved features used by rules")
        for k, v in snap.items(): print(f"  {k}: {v}")
        map_v = snap.get("map"); spo2_v = snap.get("spo2")
        if map_v is not None and -5 < map_v < 5:
            print("[WARN] MAP looks standardized (z-score). Feed UNscaled features if you expect threshold rules.")
        if spo2_v is not None and -5 < spo2_v < 5:
            print("[WARN] SpO2 looks standardized (z-score).")

    fired = evaluate_rules(x, names)
    p_ns  = combine_neuro_symbolic(p_ml, fired, alpha=args.alpha)

    prec = args.print_prec
    print("\n=== Neuro-Symbolic Mortality Risk (Factual) ===")
    print(f"Neural (ML) prob: {p_ml:.{prec}f}")
    for line in explain_rules(fired): print(" -", line)
    if not fired: print(" - No symbolic rules fired.")
    print(f"Combined neuro-symbolic prob: {p_ns:.{prec}f}")
    print("Causal DAG paths (direct):", dag_paths_to_mortality())

    # ----- single scenario via --do (unchanged) -----
    if args.do:
        raw_edits = _parse_kv_edits(args.do)
        edits: Dict[str, Any] = {}
        for k, v in raw_edits.items():
            kk = k if k in names else _resolve_concept_to_model_name(names, k)
            if kk is None:
                print(f"[warn] --do key '{k}' not found; skipping.")
                continue
            edits[kk] = v

        x_cf, desc = apply_do_intervention(x, names, edits)
        p_ml_cf = float(iso.transform(booster.predict(x_cf.reshape(1,-1)))[0])
        fired_cf = evaluate_rules(x_cf, names)
        p_ns_cf  = combine_neuro_symbolic(p_ml_cf, fired_cf, alpha=args.alpha)

        print("\n=== Counterfactual (Intervention) ===")
        print("Intervention:", desc)
        for line in explain_rules(fired_cf): print(" -", line)
        if not fired_cf: print(" - No symbolic rules fired.")
        print(f"Neural prob (cf): {p_ml_cf:.{prec}f}  |  Neuro-symbolic prob (cf): {p_ns_cf:.{prec}f}")
        print(f"Δ prob (NS): {p_ns_cf - p_ns:+.{prec}f}")

    # ----- sweeps and multi-med scenarios -----
    sweep_rows = []

    # 1) Numeric ranges (exact name or concept; optional family edit)
    for rng in args.sweep_range:
        try:
            fname_raw, values = _parse_range(rng)
        except Exception as e:
            print(f"[warn] bad --sweep-range '{rng}': {e}"); continue

        target = fname_raw if fname_raw in names else _resolve_concept_to_model_name(names, fname_raw)
        if not target:
            print(f"[warn] {fname_raw} not in model features; skipping."); continue

        for val in values:
            x_cf = _apply_family_edit(x, names, fname_raw, float(val), also_z=True) if args.edit_family \
                   else _apply_edits_vec(x, names, {target: float(val)})
            p_ml_cf = float(iso.transform(booster.predict(x_cf.reshape(1,-1)))[0])
            fired_cf = evaluate_rules(x_cf, names)
            p_ns_cf  = combine_neuro_symbolic(p_ml_cf, fired_cf, alpha=args.alpha)
            sweep_rows.append({"type":"range","feature":target,"value":float(val),
                               "ml":float(p_ml_cf),"ns":float(p_ns_cf),"fused":float(p_ns_cf),
                               "d_ml":float(p_ml_cf-p_ml),"d_ns":float(p_ns_cf-p_ns),"d_fused":float(p_ns_cf-p_ns)})

    # 2) Med sweeps (legacy quick toggles)
    if args.sweep_meds:
        meds = _med_columns(names)
        if args.sweep_meds_include:
            allow = set(m.strip() for m in args.sweep_meds_include.split(","))
            meds = [m for m in meds if m in allow]
        if not meds:
            print("[warn] no medication columns matched for sweep.")
        else:
            if args.sweep_meds in ("alloff","allon"):
                val = 0.0 if args.sweep_meds == "alloff" else 1.0
                x_cf = _apply_edits_vec(x, names, {m: val for m in meds})
                p_ml_cf = float(iso.transform(booster.predict(x_cf.reshape(1,-1)))[0])
                fired_cf = evaluate_rules(x_cf, names)
                p_ns_cf  = combine_neuro_symbolic(p_ml_cf, fired_cf, alpha=args.alpha)
                sweep_rows.append({"type":args.sweep_meds,"feature":"*","value":float(val),
                                   "ml":float(p_ml_cf),"ns":float(p_ns_cf),"fused":float(p_ns_cf),
                                   "d_ml":float(p_ml_cf-p_ml),"d_ns":float(p_ns_cf-p_ns),"d_fused":float(p_ns_cf-p_ns)})
            else:
                val = 0.0 if args.sweep_meds == "eachoff" else 1.0
                for m in meds:
                    x_cf = _apply_edits_vec(x, names, {m: val})
                    p_ml_cf = float(iso.transform(booster.predict(x_cf.reshape(1,-1)))[0])
                    fired_cf = evaluate_rules(x_cf, names)
                    p_ns_cf  = combine_neuro_symbolic(p_ml_cf, fired_cf, alpha=args.alpha)
                    sweep_rows.append({"type":args.sweep_meds,"feature":m,"value":float(val),
                                       "ml":float(p_ml_cf),"ns":float(p_ns_cf),"fused":float(p_ns_cf),
                                       "d_ml":float(p_ml_cf-p_ml),"d_ns":float(p_ns_cf-p_ns),"d_fused":float(p_ns_cf-p_ns)})

    # 3) NEW: explicit multi-med scenarios (user-defined)
    med_scenarios: List[Tuple[str, Dict[str, float]]] = []
    for s in args.med_scenario:
        try:
            label, edits = _parse_med_scenario(s)
        except Exception as e:
            print(f"[warn] bad --med-scenario '{s}': {e}"); 
            continue
        med_scenarios.append((label, edits))

    # 4) NEW: auto-generated med grid (all combinations)
    if args.med_grid:
        grid_meds = [m.strip() for m in args.med_grid.split(",") if m.strip()]
        grid_vals = [float(v) for v in args.grid_values.split(",") if v.strip()]
        med_scenarios += _generate_med_grid(names, grid_meds, grid_vals, args.grid_max)

    # Evaluate multi-med scenarios
    if med_scenarios:
        print("\n=== Medication Scenarios ===")
        for label, edits in med_scenarios:
            # ignore non-med keys silently here (this section is med-focused)
            med_edits = {k: v for k, v in edits.items() if k in names and k.startswith("med__")}
            if not med_edits:
                print(f" - {label}: [warn] no valid med__* columns; skipped")
                continue
            x_cf = _apply_edits_vec(x, names, med_edits)
            p_ml_cf = float(iso.transform(booster.predict(x_cf.reshape(1,-1)))[0])
            fired_cf = evaluate_rules(x_cf, names)
            p_ns_cf  = combine_neuro_symbolic(p_ml_cf, fired_cf, alpha=args.alpha)
            delta = float(p_ns_cf - p_ns)
            print(f" - {label}: fused {p_ns_cf:.{prec}f} (Δ {delta:+.{prec}f})")
            sweep_rows.append({"type":"med_scenario","feature":label,"value":np.nan,
                               "ml":float(p_ml_cf),"ns":float(p_ns_cf),"fused":float(p_ns_cf),
                               "d_ml":float(p_ml_cf-p_ml),"d_ns":delta,"d_fused":delta})

    if sweep_rows:
        print("\n=== Counterfactual Sweep ===")
        shown = 0
        for r in sweep_rows[:10]:
            val_str = "NA" if isinstance(r['value'], float) and np.isnan(r['value']) else str(r['value'])
            print(f" - {r['type']:>12} {r['feature']}: {val_str}  fused {r['fused']:.{prec}f} (Δ {r['d_fused']:+.{prec}f})")
            shown += 1
        if len(sweep_rows) > shown:
            print(f"   ... {len(sweep_rows) - shown} more rows")
        if args.sweep_out:
            import pandas as pd
            os.makedirs(Path(args.sweep_out).parent, exist_ok=True)
            pd.DataFrame(sweep_rows).to_csv(args.sweep_out, index=False)
            print(f"[ok] Sweep results saved → {args.sweep_out}")

if __name__ == "__main__":
    main()
