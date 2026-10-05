"""Patient-grouped train/val/test split and the 20-week screening metrics (no TensorFlow needed).

HC18 does not ship patient IDs, but repeat scans of one pregnancy can be recognised (see patient_ids):
a repeat suffix in the name and neighbouring image numbers with the same pixel size. Splitting by
those groups keeps the scans of a pregnancy on one side, so the test set cannot contain a scan of
a pregnancy the model trained on. It is a conservative heuristic, not ground truth.
"""
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .growth import hadlock_ga_weeks

THRESHOLD_WEEKS = 20.0


_NAME = re.compile(r"^(\d+)_(\d*)HC", re.IGNORECASE)


def patient_ids(filenames, pixel_mm=None, px_tol: float = 0.005) -> list[str]:
    """Group scans that probably belong to the same pregnancy. HC18 ships no patient IDs.

    Evidence in the HC18 CSV: names look like 8_2HC.png / 22_HC.png, the leading number is a running
    image number (unique per file), a repeat scan carries a 2HC/3HC/4HC suffix, and repeat scans sit
    next to their partner scans (often with the same pixel size). So scans are linked into one group when
      * they share the leading number (001_HC.png / 001_2HC.png style names), or
      * they are neighbouring image numbers and either one is a repeat scan (suffix 2 or more), or
      * they are neighbouring and have the same pixel size (within `px_tol`, relative).
    Groups are the connected chains. This errs on the side of merging, which only makes the split
    stricter. Names that do not match the pattern each stand alone."""
    names = [str(f).replace("\\", "/").rsplit("/", 1)[-1] for f in filenames]
    n = len(names)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    parsed = []
    for i, nm in enumerate(names):
        mt = _NAME.match(nm)
        parsed.append((int(mt.group(1)), mt.group(2) != "" and int(mt.group(2)) >= 2) if mt else None)

    first_with_prefix = {}
    for i, pr in enumerate(parsed):
        if pr is None:
            continue
        if pr[0] in first_with_prefix:
            union(i, first_with_prefix[pr[0]])
        else:
            first_with_prefix[pr[0]] = i
    order = sorted((i for i, pr in enumerate(parsed) if pr is not None), key=lambda i: parsed[i][0])
    px = None if pixel_mm is None else np.asarray(pixel_mm, dtype=np.float64)
    for a, b in zip(order[:-1], order[1:]):
        if parsed[b][0] - parsed[a][0] != 1:
            continue
        near = px is not None and abs(px[a] - px[b]) <= px_tol * max(px[a], px[b])
        if parsed[a][1] or parsed[b][1] or near:
            union(a, b)
    return [f"g{find(i)}" for i in range(n)]


def _carve(df: pd.DataFrame, idx: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Positions (from idx) of about `fraction` of the rows, whole patients, class mix kept."""
    n_splits = max(2, int(round(1.0 / fraction)))
    sub = df.iloc[idx]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, held = next(sgkf.split(sub, sub["label"], groups=sub["patient_id"]))
    return idx[held]


def grouped_split(df: pd.DataFrame, test_fraction: float, val_fraction: float, seed: int):
    """Add `patient_id` and `split` (train/val/test) columns; return (df, info).

    Raises AssertionError if any patient appears in two splits."""
    df = df.copy().reset_index(drop=True)
    df["patient_id"] = patient_ids(df["filename"], df["pixel_mm"] if "pixel_mm" in df else None)
    everyone = np.arange(len(df))
    te = _carve(df, everyone, test_fraction, seed)
    rest = np.setdiff1d(everyone, te)
    va = _carve(df, rest, val_fraction, seed)
    df["split"] = "train"
    df.loc[va, "split"] = "val"
    df.loc[te, "split"] = "test"

    seen = {s: set(df.loc[df["split"] == s, "patient_id"]) for s in ("train", "val", "test")}
    overlap = (seen["train"] & seen["val"]) | (seen["train"] & seen["test"]) | (seen["val"] & seen["test"])
    assert not overlap, f"patients on both sides of a split: {sorted(overlap)[:5]}"
    info = {
        "grouped_by": "groups of linked scans (repeat-scan suffix, neighbouring image numbers, same pixel size); heuristic",
        "n_images": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()),
        "images_per_split": {s: int((df["split"] == s).sum()) for s in seen},
        "patients_per_split": {s: len(v) for s, v in seen.items()},
        "patients_in_more_than_one_split": 0,
    }
    return df, info


def _counts(pos: np.ndarray, pred: np.ndarray):
    tp = int((pos & pred).sum())
    fn = int((pos & ~pred).sum())
    fp = int((~pos & pred).sum())
    tn = int((~pos & ~pred).sum())
    return tp, fp, fn, tn


def _ratios(tp, fp, fn, tn) -> dict:
    def r(a, b):
        return a / b if b else float("nan")
    return {"recall_sensitivity": r(tp, tp + fn), "precision": r(tp, tp + fp),
            "specificity": r(tn, tn + fp), "accuracy": r(tp + tn, tp + fp + fn + tn)}


def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def screening_20w(hc_true, hc_pred, groups=None, threshold: float = THRESHOLD_WEEKS,
                  n_boot: int = 2000, seed: int = 42) -> dict:
    """Recall, precision, specificity and accuracy for 'is the pregnancy `threshold` weeks or more?'.

    Both the reference and the prediction are head circumference converted to weeks with the
    Hadlock formula, the same rule the app uses. 95% intervals come from a bootstrap that
    resamples whole patients when `groups` is given, otherwise single scans."""
    ga_true = hadlock_ga_weeks(np.asarray(hc_true, dtype=np.float64))
    ga_pred = hadlock_ga_weeks(np.asarray(hc_pred, dtype=np.float64))
    pos, pred = ga_true >= threshold, ga_pred >= threshold
    tp, fp, fn, tn = _counts(pos, pred)
    point = _ratios(tp, fp, fn, tn)

    n = len(pos)
    if groups is None:
        members = [np.array([i]) for i in range(n)]
    else:
        g = np.asarray(groups)
        members = [np.flatnonzero(g == u) for u in np.unique(g)]
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in point}
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        rows = np.concatenate([members[i] for i in pick])
        for k, v in _ratios(*_counts(pos[rows], pred[rows])).items():
            boots[k].append(v)

    out = {
        "threshold_weeks": float(threshold),
        "n": int(n), "n_positive": int(pos.sum()), "n_negative": int((~pos).sum()),
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
    for k, v in point.items():
        arr = np.array(boots[k], dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        ci = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))] if len(arr) else [None, None]
        out[k] = {"value": _num(v), "ci95": ci}
    prevalence = float(pos.mean()) if n else float("nan")
    out["prevalence_at_or_above_threshold"] = _num(prevalence)
    out["majority_baseline_accuracy"] = _num(max(prevalence, 1 - prevalence))
    out["bootstrap"] = {"resamples": n_boot, "unit": "patient" if groups is not None else "scan"}
    return out
