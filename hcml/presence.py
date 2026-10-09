"""Is a fetal head in the image at all?

The age model and the input check (validity.py) both assume a fetus is present. A scan of someone who is
not pregnant (empty uterus, ovaries, bladder, other organs) is still an ultrasound, so it can sit close to
the training scans and still receive a gestational age. This module adds the missing question: a small
logistic regression on the same 2048-d ResNet50 embedding the app already computes.

Positives are the HC18 training embeddings (artifacts/ood_reference.npz). Negatives are ultrasound images
with no fetal head: real scans of non-pregnant people when supplied (python scripts/train_presence.py
--negatives DIR), plus head-free proxies made by erasing the head from HC18 scans.
"""
import numpy as np


def _unit(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


class Presence:
    def __init__(self, w: np.ndarray, b: float, threshold: float):
        self.w = np.asarray(w, np.float32).reshape(-1)
        self.b = float(b)
        self.threshold = float(threshold)

    def prob(self, emb: np.ndarray) -> float:
        """Probability that the embedding comes from a scan with a fetal head."""
        s = float(_unit(emb).reshape(-1) @ self.w + self.b)
        return float(1.0 / (1.0 + np.exp(-np.clip(s, -50, 50))))

    def present(self, emb: np.ndarray) -> tuple[bool, float]:
        p = self.prob(emb)
        return p >= self.threshold, p

    def save(self, path):
        np.savez_compressed(path, w=self.w, b=self.b, threshold=self.threshold)

    @classmethod
    def load(cls, path) -> "Presence":
        z = np.load(path)
        return cls(z["w"], float(z["b"]), float(z["threshold"]))


def fit_presence(pos: np.ndarray, neg: np.ndarray, target_pass: float = 0.98, folds: int = 5, seed: int = 0):
    """Fit the classifier and pick the threshold so that `target_pass` of cross-validated positives pass.

    Returns (Presence, info). `neg` should hold several variants per source image; folds are split by row,
    so the reported negative rate is optimistic for sources the model has seen. Hold out sources to test.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    pos, neg = _unit(pos), _unit(neg)
    reps = max(1, len(pos) // max(len(neg), 1) // 4)       # keep negatives from being drowned out
    X = np.vstack([pos, np.repeat(neg, reps, axis=0)])
    y = np.r_[np.ones(len(pos)), np.zeros(len(X) - len(pos))]

    def make():
        return LogisticRegression(C=10, max_iter=3000, class_weight="balanced")

    cv_pos = np.zeros(len(pos))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        clf = make().fit(X[tr], y[tr])
        te_pos = te[te < len(pos)]
        cv_pos[te_pos] = clf.predict_proba(X[te_pos])[:, 1]
    thr = float(min(0.5, np.percentile(cv_pos, 100 * (1 - target_pass))))
    clf = make().fit(X, y)
    model = Presence(clf.coef_[0], clf.intercept_[0], thr)
    info = {"threshold": thr, "cv_positive_pass_rate": float((cv_pos >= thr).mean()),
            "cv_positive_p01": float(np.percentile(cv_pos, 1)), "n_positive": int(len(pos)),
            "n_negative_rows": int(len(neg)), "target_pass": target_pass}
    return model, info
