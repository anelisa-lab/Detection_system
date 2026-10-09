"""Montage of HC18 head scans that the new head check refuses (variant A, primary threshold), with their scores.
Study code only; needs head_check_fit.py and the HC18 embeddings from head_check_extract.py.

    python scripts/head_check_hc18_examples.py --out eval_head_check
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd

from head_check_common import CACHE
from head_check_final import montage, score
from hcml.validity import Reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_head_check")
    args = ap.parse_args()
    out = Path(args.out)
    z = np.load(out / "head_classifier_A_with_other.npz")
    thr = float(z["threshold_95"])
    ref = Reference.load("artifacts/ood_reference.npz")
    rows = []
    for name, lst in (("hc18_testset", "hc18_testset"), ("hc18_train", "hc18_train")):
        idx = pd.read_csv(CACHE / f"{lst}_index.csv")
        emb = np.load(CACHE / f"{lst}_emb.npy")
        s = score(z, emb)
        d = np.array([ref.distance(e) for e in emb])
        idx["score"], idx["dist"], idx["set"] = s, d, name
        rows.append(idx)
    df = pd.concat(rows)
    ts = df[(df["set"] == "hc18_testset") & (df["score"] < thr)]
    print(f"test_set scans refused: {len(ts)}; score range of refused {ts.score.min():+.2f}..{ts.score.max():+.2f}")
    pick = ts.sort_values("score").iloc[np.linspace(0, len(ts) - 1, 12).astype(int)]
    montage([{"path": r["path"], "lines": ["HC18 head scan (unlabeled test_set)", f"new score {r['score']:+.2f} (thr {thr:+.2f}) REFUSED",
                                           f"current d={r['dist']:.3f} accepted"]} for _, r in pick.iterrows()],
            out / "examples_hc18_heads_refused.png", "NEW check: HC18 head scans it wrongly refuses (spread across scores)")
    df.to_csv(out / "hc18_scores.csv", index=False, columns=["image", "set", "score", "dist"])


if __name__ == "__main__":
    main()
