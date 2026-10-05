"""HC18 loading, gestational-stage labels and annotation-derived features."""
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import CLASS_NAMES
from .growth import hadlock_ga_weeks

CSV_NAME = "training_set_pixel_size_and_HC.csv"


def find_csv(data_dir: Path) -> Path:
    for p in [data_dir / CSV_NAME, data_dir.parent / CSV_NAME, *data_dir.rglob(CSV_NAME)]:
        if p.exists():
            return p
    raise FileNotFoundError(f"{CSV_NAME} not found in or next to {data_dir}")


def load_hc18(data_dir) -> pd.DataFrame:
    """Read the HC18 training CSV and attach image/annotation paths."""
    data_dir = Path(data_dir)
    df = pd.read_csv(find_csv(data_dir))
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"pixel size(mm)": "pixel_mm", "pixel size": "pixel_mm", "head circumference (mm)": "hc_mm"})
    df["image_path"] = df["filename"].map(lambda f: str(data_dir / f))
    df["annotation_path"] = df["filename"].map(
        lambda f: str(data_dir / f.replace(".png", "_Annotation.png")))
    missing = ~df["image_path"].map(lambda p: Path(p).exists())
    if missing.all():
        raise FileNotFoundError(f"No HC18 images found in {data_dir}")
    if missing.any():
        print(f"warning: {missing.sum()} images listed in the CSV are missing; skipping them")
    return df[~missing].reset_index(drop=True)


def stage_label(hc_mm: float, thresholds) -> int:
    """0=early, 1=mid, 2=late by head circumference."""
    lo, hi = thresholds
    return 0 if hc_mm < lo else (1 if hc_mm < hi else 2)


def gestational_age_weeks(hc_mm: float) -> float:
    """Hadlock (1984) gestational age from head circumference (see growth.py)."""
    return float(hadlock_ga_weeks(hc_mm))


def add_labels(df: pd.DataFrame, thresholds) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["hc_mm"].map(lambda v: stage_label(v, thresholds))
    df["stage"] = df["label"].map(lambda i: CLASS_NAMES[i])
    df["ga_weeks_hadlock"] = df["hc_mm"].map(gestational_age_weeks).round(1)
    return df


def _entropy(values: np.ndarray) -> float:
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def annotation_features(image_path: str, annotation_path: str, pixel_mm: float, hc_mm: float) -> dict:
    """Section 3.3 features: HC, ellipse area/perimeter, intensity mean/std, entropy.

    These describe the data; they are NOT model inputs, because the labels are
    built from head circumference (Section 5 explains the leakage).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    out = {"hc_mm": hc_mm, "image_entropy": _entropy(img)}
    ann = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE) if Path(annotation_path).exists() else None
    pts = cv2.findNonZero(ann) if ann is not None else None
    if pts is None or len(pts) < 5:
        return out
    (cx, cy), (w, h), angle = cv2.fitEllipse(pts)
    a, b = w / 2.0, h / 2.0
    perim_px = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))  # Ramanujan
    mask = np.zeros_like(img)
    cv2.ellipse(mask, ((cx, cy), (w, h), angle), 255, -1)
    inside = img[mask > 0]
    out.update({
        "ellipse_area_mm2": math.pi * a * b * pixel_mm ** 2,
        "ellipse_perimeter_mm": perim_px * pixel_mm,
        "mean_intensity": float(inside.mean()),
        "std_intensity": float(inside.std()),
        "ellipse_entropy": _entropy(inside),
    })
    return out


def annotation_mask(annotation_path: str, size: int) -> np.ndarray:
    """Filled head ellipse (float 0..1, size x size) from an HC18 annotation, or zeros."""
    ann = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE) if Path(annotation_path).exists() else None
    pts = cv2.findNonZero(ann) if ann is not None else None
    if pts is None or len(pts) < 5:
        return np.zeros((size, size), np.float32)
    m = np.zeros_like(ann)
    cv2.ellipse(m, cv2.fitEllipse(pts), 255, -1)
    return cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
