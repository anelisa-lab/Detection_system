"""Load the trained model and turn an uploaded file into a checked estimate."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import keras
import numpy as np

from .growth import hadlock_ga_weeks
from .model import Analyzer
from .preprocess import clean, read_gray, to_model_input
from .skull import black_boxes, boxes_cover, cam_overlap, ellipse_mask, peak_inside, skull_ellipse
from .validity import Reference

EDGE_WEEKS = (14.0, 36.0)       # training data is sparse outside this band
EDGE_SCALE = 1.5

UNUSUAL = ("This image looks unusual compared with the training scans, so the range is widened.")
LIMITED_WARNING = ("This scan is outside the standard head circumference view. "
                   "Treat the estimate as rough.")
REJECTED = ("This does not look like a standard head circumference view. The estimate would be unreliable.")


@dataclass
class Result:
    gray: np.ndarray
    cam: np.ndarray | None
    probs: np.ndarray
    hc_mm: float
    ga_weeks: float | None          # None when the input check fails
    level: str                      # good | reduced | rejected
    distance: float
    half_days: float                # likely range, +/- days
    notes: list[str] = field(default_factory=list)
    badge: str = "Rejected"         # Good | Limited | Poor | Rejected
    badge_reasons: list[str] = field(default_factory=list)
    overlap: float | None = None    # share of the heatmap lying on the detected skull
    ellipse: tuple | None = None    # detected skull ellipse (cv2 format) or None
    original: np.ndarray | None = None   # the upload as grayscale, at most 640 px on the long side
    orig_size: tuple[int, int] = (0, 0)  # (width, height) of the upload


class Predictor:
    def __init__(self, model_dir):
        d = Path(model_dir)
        self.model = keras.models.load_model(d / "model.keras", compile=False)
        self.meta = json.loads((d / "metadata.json").read_text())
        self.reg = self.meta["regression"]
        self.ref = Reference.load(d / "ood_reference.npz")
        self.locator = keras.models.load_model(d / "locator.keras", compile=False)
        self.analyzer = Analyzer(self.model, self.locator)
        self.heat = self.meta["heatmap_check"]

    def predict(self, data: bytes, name: str = "", strictness: float = 1.0) -> Result:
        if not data:
            raise ValueError("the file is empty")
        raw = read_gray(data, name)
        scale = 640 / max(raw.shape)
        original = cv2.resize(raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else raw
        size = (raw.shape[1], raw.shape[0])
        raw_small = clean(raw, denoise=False)        # exact black boxes survive the resize
        gray = clean(raw, denoise=self.meta.get("denoise", True))
        x = to_model_input(gray)
        cam, probs, z, emb, head_prob = self.analyzer(x)
        dist = self.ref.distance(emb)
        level = self.ref.level(dist, strictness)
        half = float(self.reg["interval_days"])
        if level == "rejected":      # no estimate and no heatmap for an input the model was not trained on
            return Result(gray, None, np.zeros(0), float("nan"), None, level, dist, half, [REJECTED],
                          original=original, orig_size=size)
        hc = float(z * self.reg["hc_std_mm"] + self.reg["hc_mean_mm"])
        ga = float(hadlock_ga_weeks(hc))
        notes = []
        ell = skull_ellipse(head_prob)
        mask = ellipse_mask(ell)
        overlap = cam_overlap(cam, mask) if ell is not None else 0.0
        reasons = []
        if ga <= self.heat["ga_p5_weeks"]:
            reasons.append(f"the estimate is in the lowest 5% of training ages (under {self.heat['ga_p5_weeks']:.0f} weeks)")
        if ell is None:
            reasons.append("no skull-like region was found in the image")
        else:
            off_peak = not peak_inside(cam, mask)
            if off_peak or overlap < self.heat["min_overlap"]:      # one reason, not two
                reasons.append(f"the heatmap is not on the skull ({overlap:.0%} of it on the skull"
                               f"{', peak outside' if off_peak else ''})")
            if boxes_cover(mask, black_boxes(raw_small)) > 0.02:
                reasons.append("a black box covers part of the head region")
        if dist > self.ref.p90:
            reasons.append("the image is unusual compared with the training scans (above their 90th percentile)")
        badge = "Good" if not reasons else ("Limited" if len(reasons) == 1 else "Poor")

        half *= self.ref.range_scale(dist)
        if level == "reduced":
            notes.append(UNUSUAL)
        if not EDGE_WEEKS[0] <= ga <= EDGE_WEEKS[1]:
            half *= EDGE_SCALE
            notes.append("The estimate is near the edge of the training data, so the range is widened.")
        return Result(gray, cam, probs, hc, ga, level, dist, half, notes, badge, reasons, overlap, ell, original, size)
