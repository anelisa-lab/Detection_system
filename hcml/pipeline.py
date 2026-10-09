"""Load the trained model and turn an uploaded file into a checked estimate."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import keras
import numpy as np

from .growth import hadlock_ga_weeks
from .headcheck import NOT_HEAD_VIEW, HeadCheck
from .model import Analyzer
from .presence import Presence
from .preprocess import clean, read_gray, to_model_input
from .skull import black_boxes, boxes_cover, cam_overlap, ellipse_mask, peak_inside, skull_ellipse
from .validity import Reference

EDGE_WEEKS = (14.0, 36.0)       # training data is sparse outside this band
EDGE_SCALE = 1.5

UNUSUAL = ("This image looks unusual compared with the training scans, so the range is widened.")
LIMITED_WARNING = ("This scan is outside the standard head circumference view. "
                   "Treat the estimate as rough.")
REJECTED = ("This does not look like a standard head circumference view. The estimate would be unreliable.")
NO_FETUS = ("No fetal head was found in this image, so no gestational age is given.")


SCAN_MAX_COLOURFULNESS = 10.0   # mean (max - min) over R, G, B; ultrasound scans are grey (about 0), photos are not


def looks_like_scan(data: bytes) -> bool:
    """True for an image that is grey like an ultrasound. Photos and other colourful pictures are not scans.
    Files OpenCV cannot decode as a picture (DICOM) count as scans."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return True
    img = img.astype(np.int16)
    return float((img.max(-1) - img.min(-1)).mean()) < SCAN_MAX_COLOURFULNESS


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
    badge: str = "Rejected"         # Good | Limited | Poor | Rejected | No fetus
    badge_reasons: list[str] = field(default_factory=list)
    overlap: float | None = None    # share of the heatmap lying on the detected skull
    ellipse: tuple | None = None    # detected skull ellipse (cv2 format) or None
    original: np.ndarray | None = None   # the upload as grayscale, at most 640 px on the long side
    orig_size: tuple[int, int] = (0, 0)  # (width, height) of the upload
    head_score: float | None = None      # head-view check score (None when that check is not installed)
    head_threshold: float | None = None  # its threshold at the operating point used for this scan
    rejected_by: str | None = None       # "head check" or "image check" when no estimate is given
    no_fetus: bool = False          # the presence check found no fetal head (e.g. a scan of someone not pregnant)
    head_present_prob: float | None = None   # presence-check probability, None when no presence model is loaded
    presence_threshold: float | None = None  # its pass threshold


class Predictor:
    def __init__(self, model_dir):
        d = Path(model_dir)
        self.model = keras.models.load_model(d / "model.keras", compile=False)
        self.meta = json.loads((d / "metadata.json").read_text())
        self.reg = self.meta["regression"]
        self.ref = Reference.load(d / "ood_reference.npz")
        self.locator = keras.models.load_model(d / "locator.keras", compile=False)
        # Older model folders have no presence check; the app then behaves as before.
        self.presence = Presence.load(d / "presence.npz") if (d / "presence.npz").exists() else None
        self.analyzer = Analyzer(self.model, self.locator)
        # The head-view check lives in its own folder; without it the app behaves as before.
        hc = d / "head_check"
        self.head = HeadCheck.load(hc) if (hc / "classifier.npz").exists() else None
        self.heat = self.meta["heatmap_check"]

    def predict(self, data: bytes, name: str = "", strictness: float = 1.0, head_target: int | None = None) -> Result:
        """`head_target` is the head-view check operating point (95 default, or 98); None uses HEAD_CHECK_TARGET/95."""
        if not data:
            raise ValueError("the file is empty")
        raw = read_gray(data, name)
        scale = 640 / max(raw.shape)
        original = cv2.resize(raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else raw
        size = (raw.shape[1], raw.shape[0])
        raw_small = clean(raw, denoise=False)        # exact black boxes survive the resize
        gray = clean(raw, denoise=self.meta.get("denoise", True))
        x = to_model_input(gray)
        cam, probs, z, emb, loc_prob = self.analyzer(x)
        dist = self.ref.distance(emb)
        level = self.ref.level(dist, strictness)
        half = float(self.reg["interval_days"])
        presence_prob = None
        if self.presence is not None:
            found, presence_prob = self.presence.present(emb)
            if not found and not looks_like_scan(data):    # a photo, not a scan: refused as before, never "no fetus"
                return Result(gray, None, np.zeros(0), float("nan"), None, "rejected", dist, half, [REJECTED],
                              original=original, orig_size=size, head_present_prob=presence_prob)
            if not found:        # nothing like a fetus: no age, no heatmap, no screening result
                return Result(gray, None, np.zeros(0), float("nan"), None, "rejected", dist, half, [NO_FETUS],
                              badge="No fetus", original=original, orig_size=size, no_fetus=True,
                              head_present_prob=presence_prob, presence_threshold=self.presence.threshold)
        head_score = head_thr = None
        head_ok = True
        if self.head is not None:     # accepted only if BOTH this check and the image check accept
            head_thr = self.head.threshold(head_target)
            head_score = self.head.score(emb)
            head_ok = head_score >= head_thr
        if not head_ok or level == "rejected":   # no estimate and no heatmap for an input the model was not trained on
            return Result(gray, None, np.zeros(0), float("nan"), None, "rejected", dist, half,
                          [NOT_HEAD_VIEW if not head_ok else REJECTED], original=original, orig_size=size,
                          head_present_prob=presence_prob, head_score=head_score, head_threshold=head_thr,
                          rejected_by="head check" if not head_ok else "image check")
        hc = float(z * self.reg["hc_std_mm"] + self.reg["hc_mean_mm"])
        ga = float(hadlock_ga_weeks(hc))
        notes = []
        ell = skull_ellipse(loc_prob)
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
        return Result(gray, cam, probs, hc, ga, level, dist, half, notes, badge, reasons, overlap, ell, original, size,
                      head_score=head_score, head_threshold=head_thr, head_present_prob=presence_prob)
