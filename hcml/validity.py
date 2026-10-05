"""Is this image a standard head-circumference view like the training data?

HC18 holds only axial fetal-head planes from about 12 to 40 weeks. A first-trimester
whole-fetus (CRL) view, a photo, or a poor frame still gets a number from a regressor,
so the model must be asked first whether the input resembles what it was trained on.

Score = mean cosine distance from the image's ResNet50 embedding to its K nearest
training embeddings. Two cut points are calibrated on held-out valid scans: 'warn'
(95th percentile: image is unusual, widen the range) and 'reject' (99.5th percentile:
no estimate). A hand-built ellipse detector was tried and dropped, because it rejected
valid HC18 scans and scored a CRL view higher than the typical valid scan.
"""
import numpy as np

K = 5


def _unit(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


class Reference:
    def __init__(self, feats: np.ndarray, warn: float, reject: float, p90: float | None = None):
        self.feats = _unit(feats)
        self.warn, self.reject = float(warn), float(reject)
        self.p90 = float(p90) if p90 is not None else self.warn

    def distance(self, emb: np.ndarray) -> float:
        sims = self.feats @ _unit(emb).reshape(-1)
        return float(1.0 - np.sort(sims)[-K:].mean())

    def level(self, d: float, strictness: float = 1.0) -> str:
        """'good', 'reduced' or 'rejected'. strictness < 1 rejects sooner."""
        if d > self.reject * strictness:
            return "rejected"
        return "reduced" if d > self.warn * min(strictness, 1.0) else "good"

    def range_scale(self, d: float) -> float:
        """1.0 up to 'warn', rising linearly to 2.0 at 'reject'."""
        return 1.0 + float(np.clip((d - self.warn) / (self.reject - self.warn), 0.0, 1.0))

    def save(self, path):
        np.savez_compressed(path, feats=self.feats.astype(np.float16), warn=self.warn, reject=self.reject, p90=self.p90)

    @classmethod
    def load(cls, path) -> "Reference":
        z = np.load(path)
        return cls(z["feats"].astype(np.float32), float(z["warn"]), float(z["reject"]),
                   float(z["p90"]) if "p90" in z else None)


def fit_reference(train_feats, heldout_feats, warn_pct=95.0, reject_pct=99.5) -> tuple[Reference, dict]:
    """Calibrate cut points on valid scans that were not used to build the reference."""
    probe = Reference(train_feats, 0, 1)
    d = np.array([probe.distance(e) for e in heldout_feats])
    ref = Reference(train_feats, np.percentile(d, warn_pct), np.percentile(d, reject_pct), np.percentile(d, 90))
    info = {"k": K, "warn": ref.warn, "p90": ref.p90, "reject": ref.reject, "heldout_n": int(len(d)),
            "heldout_median": float(np.median(d)), "heldout_max": float(d.max()),
            "expected_false_reject_rate": round(1 - reject_pct / 100, 4)}
    return ref, info
