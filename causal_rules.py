from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import math
import numpy as np

@dataclass
class FiredRule:
    name: str
    evidence: List[str]       
    weight: float             
    details: Dict[str, Any]  

ALIASES: Dict[str, Tuple[str, ...]] = {
    "map":        ("map", "mbp", "mean_bp", "mean arterial", "mean blood pressure"),
    "hr":         ("heart_rate", "hr"),
    "rr":         ("resp_rate", "rr", "respiratory"),
    "spo2":       ("spo2", "sao2", "o2sat", "oxygen saturation"),
    "temp":       ("temp", "temperature"),
    "wbc":        ("wbc", "white blood cell"),
    "lactate":    ("lactate",),
    "creatinine": ("creatinine", "cr"),
    "bun":        ("bun", "urea"),
    "gcs":        ("gcs", "glasgow"),
    # medication flags (pooled to patient-level): any-hour exposure in first 24h -> 1
    "med_vaso":   ("med__pressor", "med__vasopressor", "vasopressor"),
    "med_sed":    ("med__sedation", "sedation", "sedative", "opioid"),
    "med_beta":    ("med__beta_blocker", "beta_blocker", "beta-blocker"),
}

# Pretty labels for output
LABELS = {
    "hypotension": "MAP < 65 mmHg",
    "tachycardia": "HR ≥ 100 bpm",
    "bradycardia": "HR ≤ 50 bpm",
    "tachypnea":   "RR ≥ 24",
    "hypoxia":     "SpO₂ < 92%",
    "fever":       "Temp ≥ 38°C",
    "hypothermia": "Temp ≤ 35°C",
    "hyperlact":   "Lactate ≥ 2 mmol/L",
    "hyperlact4":  "Lactate ≥ 4 mmol/L",
    "aki":         "Creatinine ≥ 1.5 mg/dL",
    "aki2":        "Creatinine ≥ 2.0 mg/dL",
    "azotemia":    "BUN ≥ 20 mg/dL",
    "leukocytosis":"WBC ≥ 12×10⁹/L",
    "leukopenia":  "WBC ≤ 4×10⁹/L",
    "low_gcs":     "GCS ≤ 8",
    "med_vaso":    "Vasopressor in 24h",
    "med_sed":     "Sedation/opiate in 24h",
    "med_beta":    "Beta-blocker in 24h",
}

def _find_index(names: List[str], candidates: Tuple[str, ...]) -> Optional[int]:
    low = [n.lower() for n in names]
    for a in candidates:
        a = a.lower()
        for i, n in enumerate(low):
            if a in n:
                return i
    return None

def _get_value(x_row: np.ndarray, names: List[str], key: str) -> Optional[float]:
    idx = _find_index(names, ALIASES[key])
    if idx is None:
        return None
    try:
        v = float(x_row[idx])
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

def _get_flag01(x_row: np.ndarray, names: List[str], key: str) -> bool:
    # Treat >= 0.5 as present
    idx = _find_index(names, ALIASES[key])
    if idx is None:
        return False
    try:
        return float(x_row[idx]) >= 0.5
    except Exception:
        return False


def _looks_standardized(val: Optional[float], expected_range=(10, 300)) -> bool:
    
    if val is None:
        return False
    if abs(val) > 6:
        return False
    return True

Z_THRESH = {
    "low":  -1.0,
    "high": +1.0,
}


def debug_snapshot(x_row: np.ndarray, names: List[str]) -> Dict[str, Any]:
    snap = {}
    for key in ("map","hr","rr","spo2","temp","wbc","lactate","creatinine","bun","gcs"):
        snap[key] = _get_value(x_row, names, key)
    snap["med_vaso_present"] = _get_flag01(x_row, names, "med_vaso")
    snap["med_sed_present"]  = _get_flag01(x_row, names, "med_sed")
    snap["med_beta_present"] = _get_flag01(x_row, names, "med_beta")
    return snap


def _severity_above(val: float, thr: float, span: float) -> float:
    """Scale 0..1 as value rises above threshold by 'span' (clipped)."""
    if val is None:
        return 0.0
    return float(np.clip((val - thr) / max(span, 1e-6), 0.0, 1.0))

def _severity_below(val: float, thr: float, span: float) -> float:
    """Scale 0..1 as value falls below threshold by 'span' (clipped)."""
    if val is None:
        return 0.0
    return float(np.clip((thr - val) / max(span, 1e-6), 0.0, 1.0))


def evaluate_rules(x_row: np.ndarray, names: List[str]) -> List[FiredRule]:
    fired: List[FiredRule] = []

    # Fetch values
    v_map  = _get_value(x_row, names, "map")
    v_hr   = _get_value(x_row, names, "hr")
    v_rr   = _get_value(x_row, names, "rr")
    v_spo2 = _get_value(x_row, names, "spo2")
    v_tmp  = _get_value(x_row, names, "temp")
    v_wbc  = _get_value(x_row, names, "wbc")
    v_lac  = _get_value(x_row, names, "lactate")
    v_cr   = _get_value(x_row, names, "creatinine")
    v_bun  = _get_value(x_row, names, "bun")
    v_gcs  = _get_value(x_row, names, "gcs")

    f_vaso = _get_flag01(x_row, names, "med_vaso")
    f_sed  = _get_flag01(x_row, names, "med_sed")
    f_beta = _get_flag01(x_row, names, "med_beta")

    use_z = False
    sentinels = [v_map, v_spo2, v_hr]
    if sum(_looks_standardized(v) for v in sentinels if v is not None) >= 2:
        use_z = True

    ev = []
    sev = 0.0
    if not use_z:
        hypo = (v_map is not None and v_map < 65)
        sev  = max(sev, _severity_below(v_map, 65, span=15.0))  # 65→50 ~ moderate→severe
    else:
        hypo = (v_map is not None and v_map < Z_THRESH["low"])  # z < -1
        # approximate severity in z-space
        sev  = max(sev, _severity_below(v_map, Z_THRESH["low"], span=1.5))

    if f_vaso:
        ev.append(LABELS["med_vaso"])
        sev = max(sev, 0.6)
    if hypo:
        ev.append(LABELS["hypotension"])

    if (f_vaso or hypo):
        # lactate gate
        hyperlact = False
        if not use_z:
            if v_lac is not None and v_lac >= 2:
                hyperlact = True
                ev.append(LABELS["hyperlact"])
                sev = max(sev, _severity_above(v_lac, 2.0, span=2.0))  # 2→4 scale
            if v_lac is not None and v_lac >= 4:
                ev.append(LABELS["hyperlact4"])
                sev = max(sev, 1.0)
        else:
            if v_lac is not None and v_lac > Z_THRESH["high"]:
                hyperlact = True
                ev.append("Lactate (z) > +1")
                sev = max(sev, _severity_above(v_lac, Z_THRESH["high"], span=1.0))

        if hyperlact:
            base_w = 1.0   # strong up-weight for shock-like state
            weight = base_w * max(0.4, sev)   # ensure minimum if fired
            fired.append(FiredRule(
                name="shock_like_state",
                evidence=ev.copy(),
                weight=+float(weight),
                details={"map": v_map, "lactate": v_lac, "vasopressor": f_vaso, "severity": round(weight, 3), "use_z": use_z}
            ))

  
    ev = []
    sev = 0.0
    if not use_z:
        hypox = (v_spo2 is not None and v_spo2 < 92)
        sev = max(sev, _severity_below(v_spo2, 92, span=8.0))   # 92→84
        tachp = (v_rr is not None and v_rr >= 24)
        sev = max(sev, _severity_above(v_rr, 24, span=10.0))    # 24→34
    else:
        hypox = (v_spo2 is not None and v_spo2 < Z_THRESH["low"])
        sev = max(sev, _severity_below(v_spo2, Z_THRESH["low"], span=1.5))
        tachp = (v_rr is not None and v_rr > Z_THRESH["high"])
        sev = max(sev, _severity_above(v_rr, Z_THRESH["high"], span=1.5))

    if hypox: ev.append(LABELS["hypoxia"])
    if tachp: ev.append(LABELS["tachypnea"])
    if hypox and tachp:
        w = 0.6 * max(0.4, sev)
        fired.append(FiredRule(
            name="respiratory_compromise",
            evidence=ev.copy(),
            weight=+float(w),
            details={"spo2": v_spo2, "rr": v_rr, "severity": round(w, 3), "use_z": use_z}
        ))


    ev = []
    sev = 0.0
    if not use_z:
        fever = (v_tmp is not None and v_tmp >= 38.0)
        sev = max(sev, _severity_above(v_tmp if v_tmp is not None else 38.0, 38.0, span=2.0))
        leukocyt = (v_wbc is not None and v_wbc >= 12.0)
        leukopen = (v_wbc is not None and v_wbc <= 4.0)
    else:
        fever = (v_tmp is not None and v_tmp > Z_THRESH["high"])
        leukocyt = (v_wbc is not None and v_wbc > Z_THRESH["high"])
        leukopen = (v_wbc is not None and v_wbc < Z_THRESH["low"])

    if fever: ev.append(LABELS["fever"])
    if leukocyt: ev.append(LABELS["leukocytosis"])
    if leukopen: ev.append(LABELS["leukopenia"])
    if fever and (leukocyt or leukopen):
        w = 0.4 * max(0.4, sev)
        fired.append(FiredRule(
            name="inflammatory_state",
            evidence=ev.copy(),
            weight=+float(w),
            details={"temp": v_tmp, "wbc": v_wbc, "severity": round(w, 3), "use_z": use_z}
        ))

   
    ev = []
    sev = 0.0
    aki = False
    if not use_z:
        if v_cr is not None and v_cr >= 1.5:
            aki = True; ev.append(LABELS["aki"]);  sev = max(sev, _severity_above(v_cr, 1.5, span=0.7))
        if v_cr is not None and v_cr >= 2.0:
            ev.append(LABELS["aki2"]); sev = max(sev, 1.0)
        if v_bun is not None and v_bun >= 20:
            ev.append(LABELS["azotemia"]); sev = max(sev, _severity_above(v_bun, 20, span=20.0))
    else:
        if v_cr is not None and v_cr > Z_THRESH["high"]:
            aki = True; ev.append("Creatinine (z) > +1"); sev = max(sev, _severity_above(v_cr, Z_THRESH["high"], span=1.0))
        if v_bun is not None and v_bun > Z_THRESH["high"]:
            ev.append("BUN (z) > +1"); sev = max(sev, _severity_above(v_bun, Z_THRESH["high"], span=1.0))

    if aki or ("azotemia" in " ".join(ev).lower()):
        w = 0.5 * max(0.4, sev)
        fired.append(FiredRule(
            name="renal_dysfunction",
            evidence=ev.copy(),
            weight=+float(w),
            details={"creatinine": v_cr, "bun": v_bun, "severity": round(w, 3), "use_z": use_z}
        ))

   
    if v_gcs is not None:
        ev = []
        if not use_z:
            if v_gcs <= 8:
                ev.append(LABELS["low_gcs"])
                sev = _severity_below(v_gcs, 8.0, span=3.0)
                w = 0.4 * max(0.4, sev)
                fired.append(FiredRule(
                    name="neurologic_depression",
                    evidence=ev.copy(),
                    weight=+float(w),
                    details={"gcs": v_gcs, "severity": round(w, 3)}
                ))
        else:
            # if standardized, treat z < -1 as low GCS proxy if such a feature exists
            if v_gcs < Z_THRESH["low"]:
                ev.append("GCS (z) < -1")
                w = 0.35
                fired.append(FiredRule(
                    name="neurologic_depression",
                    evidence=ev.copy(),
                    weight=+float(w),
                    details={"gcs_z": v_gcs, "severity": round(w, 3)}
                ))

   
    if f_beta and v_hr is not None:
        no_tachy = (v_hr < 100) if not use_z else (v_hr < Z_THRESH["high"])
        if no_tachy:
            fired.append(FiredRule(
                name="rate_controlled",
                evidence=[LABELS["med_beta"]],
                weight=-0.2,   # small downward nudge
                details={"beta_blocker": True, "hr": v_hr, "use_z": use_z}
            ))

   
    if not use_z:
        if v_map is not None and v_map < 55:
            fired.append(FiredRule(
                name="extreme_hypotension",
                evidence=[f"MAP < 55 (MAP={v_map:.1f})"],
                weight=+0.4,
                details={"map": v_map}
            ))
        if v_lac is not None and v_lac >= 6:
            fired.append(FiredRule(
                name="extreme_lactate",
                evidence=[f"Lactate ≥ 6 (Lac={v_lac:.1f})"],
                weight=+0.4,
                details={"lactate": v_lac}
            ))
    else:
        if v_map is not None and v_map < (Z_THRESH["low"] - 0.5):
            fired.append(FiredRule(
                name="extreme_hypotension_z",
                evidence=[f"MAP (z) < {Z_THRESH['low']-0.5:.1f}"],
                weight=+0.35,
                details={"map_z": v_map}
            ))
        if v_lac is not None and v_lac > (Z_THRESH["high"] + 1.0):
            fired.append(FiredRule(
                name="extreme_lactate_z",
                evidence=[f"Lactate (z) > {Z_THRESH['high']+1.0:.1f}"],
                weight=+0.35,
                details={"lactate_z": v_lac}
            ))

    return fired

RULE_SET_DOC = """
Rule family and conditions (units; z-score fallbacks used if standardized inputs are detected):

1) Shock-like state (+1.0 × severity)
   - (Vasopressor present OR MAP < 65) AND Lactate ≥ 2 (≥4 escalates severity)
2) Respiratory compromise (+0.6 × severity)
   - SpO₂ < 92% AND RR ≥ 24
3) Inflammatory state (+0.4 × severity)
   - Fever (Temp ≥ 38°C) AND (WBC ≥ 12 or WBC ≤ 4)
4) Renal dysfunction (+0.5 × severity)
   - Creatinine ≥ 1.5 (≥2 escalates) and/or BUN ≥ 20
5) Neurologic depression (+0.4 × severity)
   - GCS ≤ 8  (if available)
6) Protective modulation (−0.2)
   - Beta-blocker present AND no tachycardia (HR < 100)

Extreme derangements (additive):
- MAP < 55: +0.4
- Lactate ≥ 6: +0.4
(z-score analogs applied if inputs appear standardized)

Notes:
- 'severity' scales with distance beyond thresholds and is clipped to [0,1].
- Weights are added in logit space by the reasoner to produce the neuro-symbolic probability.
"""
