"""Shared helpers for the 'is this a fetal head?' check study (FETAL_PLANES_DB). Study code only: nothing here is used by
the app, and nothing in artifacts/ is changed."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Where the image lists and embeddings live (written by head_check_make_lists.py / head_check_extract.py).
CACHE = Path(os.environ.get("HEAD_CHECK_CACHE", "/home/user/head_check_cache"))
HEAD_PLANE = "Fetal brain"
THALAMIC = "Trans-thalamic"
RECALL_TARGETS = (0.90, 0.95, 0.98)       # head-acceptance targets used to set the threshold on validation
PRIMARY_TARGET = 0.95                      # fixed before any test image was scored


def load_fetal(cache: Path = CACHE):
    if not (Path(cache) / "fetal_emb.npy").exists():
        raise FileNotFoundError(
            f"No embeddings in {cache}. Run scripts/head_check_make_lists.py and scripts/head_check_extract.py first "
            "(commands: project_package/04_code_and_models/README.txt), or set HEAD_CHECK_CACHE to the folder that holds them.")
    df = pd.read_csv(cache / "fetal_index.csv")
    emb = np.load(cache / "fetal_emb.npy")
    df["cls"] = df["Plane"].str.strip()
    df["subplane"] = df["Brain_plane"].astype(str).str.strip()
    df["head"] = df["cls"] == HEAD_PLANE
    df["is_other"] = df["cls"] == "Other"
    df["thalamic"] = df["head"] & (df["subplane"] == THALAMIC)
    df["machine"] = df["US_Machine"].astype(str).str.strip()
    df["patient"] = df["Patient_num"].astype(int)
    return df, emb


def patient_split(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """train / val / test by patient (about 60 / 20 / 20), class mix kept. No patient is in two splits."""
    idx = np.arange(len(df))
    y, g = df["head"].to_numpy(), df["patient"].to_numpy()
    _, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(idx, y, g))
    rest = np.setdiff1d(idx, te)
    _, va_rel = next(StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(rest, y[rest], g[rest]))
    va = rest[va_rel]
    split = pd.Series("train", index=df.index)
    split.iloc[va] = "val"
    split.iloc[te] = "test"
    seen = {s: set(df.loc[split == s, "patient"]) for s in ("train", "val", "test")}
    assert not (seen["train"] & seen["val"] or seen["train"] & seen["test"] or seen["val"] & seen["test"]), "patient leak"
    return split


def threshold_for_recall(head_scores: np.ndarray, target: float) -> float:
    """Highest threshold that still accepts at least `target` of the head images (accept if score >= threshold)."""
    return float(np.quantile(head_scores, 1.0 - target, method="lower"))


def boot_ci(values, patients, stat=np.mean, n_boot=2000, seed=0):
    """95% interval of stat(values), resampling whole patients."""
    values, patients = np.asarray(values, dtype=float), np.asarray(patients)
    if len(values) == 0:
        return [None, None]
    groups = [np.flatnonzero(patients == p) for p in np.unique(patients)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        v = stat(values[idx])
        if v == v:
            out.append(v)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if out else [None, None]


def rate(mask, patients) -> dict:
    """Share of True in mask with a patient-bootstrap interval."""
    mask = np.asarray(mask, dtype=float)
    return {"n": int(len(mask)), "k": int(mask.sum()), "value": float(mask.mean()) if len(mask) else None,
            "ci95": boot_ci(mask, patients)}


def auc(scores_pos, scores_neg) -> float:
    """Probability a random head scores above a random non-head (ties count half)."""
    from scipy.stats import rankdata
    s = np.concatenate([scores_pos, scores_neg])
    r = rankdata(s)[: len(scores_pos)]
    return float((r.sum() - len(scores_pos) * (len(scores_pos) + 1) / 2) / (len(scores_pos) * len(scores_neg)))
