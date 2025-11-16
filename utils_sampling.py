# utils_sampling.py
import numpy as np

def make_one_to_one_sample(X, y, random_state: int = 42):
    """
    Keep ALL demises (y==1). Randomly choose the same number of survivors (y==0).
    Returns (X_bal, y_bal, kept_indices)
    """
    y = np.asarray(y).astype(int)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos = len(pos_idx)
    if n_pos == 0:
        raise ValueError("No positive (demise) cases found.")

    rng = np.random.default_rng(random_state)
    if len(neg_idx) >= n_pos:
        neg_keep = rng.choice(neg_idx, size=n_pos, replace=False)
        keep = np.concatenate([pos_idx, neg_keep])
    else:
        # rare: more deaths than survivors — downsample positives
        pos_keep = rng.choice(pos_idx, size=len(neg_idx), replace=False)
        keep = np.concatenate([pos_keep, neg_idx])

    rng.shuffle(keep)
    return X[keep], y[keep], keep
