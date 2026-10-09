"""Head-view check: does this image look like a standard fetal head ultrasound?

A logistic regression on the 2048-d embedding of the app's own frozen ResNet50 encoder (the same embedding the
embedding-distance image check uses). It was fitted on FETAL_PLANES_DB (fetal brain planes vs abdomen, femur, thorax,
maternal cervix and other planes) plus HC18 head scans, and its threshold was chosen on validation data only.

Its files are separate from the age model and from the existing image-check thresholds:

    artifacts/head_check/classifier.npz     mean, scale, coef, intercept (the standardiser and the regression)
    artifacts/head_check/thresholds.json    score thresholds for each operating point, and where they came from

The operating point is the share of validation head images the check is set to accept: 95 (default) or 98 (accepts
more heads, lets a few more non-head images through) or 90. Change it with the HEAD_CHECK_TARGET environment variable
or the Head-view check setting in the app; no code change is needed. The app accepts an image only if BOTH this
check and the existing embedding-distance check accept it.

Study code and results: scripts/head_check_*.py and eval_head_check*/.
"""
import json
import os
from pathlib import Path

import numpy as np

ENV_VAR = "HEAD_CHECK_TARGET"
DEFAULT_TARGET = 95
NOT_HEAD_VIEW = "This image does not look like a standard fetal head view, so no estimate is given."
NOT_HEAD_REASON = "the image does not look like a standard fetal head view"


class HeadCheck:
    def __init__(self, mean, scale, coef, intercept, thresholds: dict, default: int = DEFAULT_TARGET, info=None):
        self.mean = np.asarray(mean, np.float32)
        self.scale = np.asarray(scale, np.float32)
        self.coef = np.asarray(coef, np.float32)
        self.intercept = float(np.asarray(intercept).reshape(-1)[0])
        self.thresholds = {int(k): float(v) for k, v in thresholds.items()}
        self.default = int(default)
        self.info = info or {}

    @classmethod
    def load(cls, directory) -> "HeadCheck":
        d = Path(directory)
        z = np.load(d / "classifier.npz")
        meta = json.loads((d / "thresholds.json").read_text())
        return cls(z["mean"], z["scale"], z["coef"], z["intercept"], meta["operating_points"],
                   meta.get("default", DEFAULT_TARGET), meta.get("provenance"))

    @property
    def targets(self) -> list[int]:
        return sorted(self.thresholds)

    def target(self, target=None) -> int:
        """The operating point to use: the argument, else HEAD_CHECK_TARGET, else the file's default (95)."""
        if target is None:
            raw = os.environ.get(ENV_VAR, "").strip()
            target = raw or self.default
        try:
            t = int(target)
        except (TypeError, ValueError):
            raise ValueError(f"{ENV_VAR} must be one of {self.targets}, got {target!r}") from None
        if t not in self.thresholds:
            raise ValueError(f"head-check operating point {t} is not available; choose one of {self.targets}")
        return t

    def threshold(self, target=None) -> float:
        return self.thresholds[self.target(target)]

    def score(self, embedding) -> float:
        """Logit that the image is a fetal head view (higher means more head-like)."""
        z = (np.asarray(embedding, np.float32).reshape(-1) - self.mean) / self.scale
        return float(z @ self.coef + self.intercept)

    def accepts(self, embedding, target=None) -> bool:
        return self.score(embedding) >= self.threshold(target)
