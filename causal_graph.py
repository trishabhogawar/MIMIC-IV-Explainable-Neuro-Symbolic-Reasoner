# causal_graph.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

# Minimal DAG: causes → effect (Mortality)
# Nodes: Shock, RespiratoryFailure, Sepsis/Inflammation, AKI → Mortality
DAG = {
    "Shock": ["Mortality"],
    "RespiratoryFailure": ["Mortality"],
    "Inflammation": ["Mortality"],
    "AKI": ["Mortality"],
}

# Map symbolic rule names to DAG nodes
RULE_TO_NODE = {
    "shock_like_state": "Shock",
    "respiratory_compromise": "RespiratoryFailure",
    "inflammatory_state": "Inflammation",
    "renal_dysfunction": "AKI",
    "rate_controlled": None,  # modulates but not a direct node here
}

def dag_paths_to_mortality() -> List[List[str]]:
    paths = []
    for src, outs in DAG.items():
        for out in outs:
            if out == "Mortality":
                paths.append([src, out])
    return paths

def apply_do_intervention(x_row, names, intervention: Dict[str, Any]) -> Tuple[Any, Any]:
    """
    Simple 'do()' operator:
      - change selected features (e.g., set vasopressor=0; increase MAP to 70)
      - return a shallow-copied new vector and a description
    """
    import numpy as np
    xr = np.array(x_row, dtype=float, copy=True)
    desc = []
    for key, value in intervention.items():
        # key is substring to find inside names
        low = [n.lower() for n in names]
        hit = None
        for i, n in enumerate(low):
            if key.lower() in n:
                hit = i; break
        if hit is not None:
            xr[hit] = float(value)
            desc.append(f"do({names[hit]}={value})")
    return xr, "; ".join(desc) if desc else "no-op"
