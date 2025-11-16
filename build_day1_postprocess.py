# build_day1_postprocess.py
import json, numpy as np
from pathlib import Path

IN_X   = Path("X_data.npy")           # (N, 24, F)
IN_FN  = Path("feature_names.json")   # list[str] length F
OUT_X  = Path("X_tab.npy")            # (N, D_tab)
OUT_FN = Path("feature_names_tab.json")

def _time_stats(X24: np.ndarray):
    """
    X24: (N, 24, F) -> dict of {name: (N, F)} for min/max/mean/last/slope/std/frac_measured/time_since_last
    Assumes missing already imputed; to approximate 'measured', we treat values equal to the global feature median
    across time as 'potentially imputed'. If you have explicit missing masks, plug them here instead.
    """
    N,T,F = X24.shape
    t = np.arange(T, dtype=np.float32)
    # core stats
    v_min  = X24.min(1)
    v_max  = X24.max(1)
    v_mean = X24.mean(1)
    v_last = X24[:, -1, :]
    # slope via normalized time
    tn = (t - t.mean()) / (t.std() + 1e-8)
    cov  = (X24 * tn[None,:,None]).sum(1) / (T - 1)
    var_t= (tn**2).sum() / (T - 1)
    v_slope = cov / (var_t + 1e-8)
    # std across time
    v_std = X24.std(1)

    # frac_measured + time_since_last (heuristic)
    # If you have explicit masks, replace with them.
    global_med = np.median(X24, axis=(0,1), keepdims=True)   # (1,1,F)
    approx_missing = (np.isclose(X24, global_med, atol=1e-6)).astype(np.float32)
    measured = 1.0 - approx_missing
    v_frac_measured = measured.mean(1)

    # time since last 'measured' (walk from end)
    tsl = np.zeros((N,F), dtype=np.float32)
    for i in range(N):
        for f in range(F):
            seen = np.where(measured[i,:,f] > 0.5)[0]
            tsl[i,f] = (T - 1 - seen.max()) if seen.size>0 else T
    return {
        "min": v_min, "max": v_max, "mean": v_mean, "last": v_last,
        "slope": v_slope, "std": v_std,
        "frac_measured": v_frac_measured, "time_since_last": tsl
    }

def main():
    X = np.load(IN_X)            # (N,24,F)
    names = json.loads(IN_FN.read_text())

    stats = _time_stats(X)
    blocks = []
    out_names = []
    for tag in ["min","max","mean","last","slope","std","frac_measured","time_since_last"]:
        blocks.append(stats[tag])
        out_names.extend([f"{n}__{tag}" for n in names])

    X_tab = np.concatenate(blocks, axis=1)  # (N, 8*F)
    np.save(OUT_X, X_tab)
    OUT_FN.write_text(json.dumps(out_names, indent=2))

    print(f"[postprocess] X_tab.npy shape={X_tab.shape}")
    print(f"[postprocess] feature_names_tab.json count={len(out_names)}")

if __name__ == "__main__":
    main()
