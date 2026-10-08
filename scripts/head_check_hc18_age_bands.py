"""How often does the new head check accept HC18 head scans, by gestational age band (Hadlock age from HC)?
Study code only.  python scripts/head_check_hc18_age_bands.py --out eval_head_check
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from head_check_common import CACHE, boot_ci
from head_check_final import score
from hcml.growth import hadlock_ga_weeks
from hcml.splitting import patient_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_head_check")
    out = Path(ap.parse_args().out)
    res = {}
    idx = pd.read_csv(CACHE / "hc18_train_index.csv")
    emb = np.load(CACHE / "hc18_train_emb.npy")
    idx["ga"] = hadlock_ga_weeks(idx["head circumference (mm)"].values)
    idx["pid"] = patient_ids(idx["filename"], idx["pixel size(mm)"])
    idx["band"] = pd.cut(idx.ga, [0, 17, 20, 24, 28, 45], labels=["<17 wk", "17-20", "20-24", "24-28", "28+"])
    for var in ("A_with_other", "B_without_other"):
        z = np.load(out / f"head_classifier_{var}.npz")
        idx["acc"] = score(z, emb) >= float(z["threshold_95"])
        r = {}
        for b, g in idx.groupby("band", observed=True):
            r[str(b)] = {"n": int(len(g)), "accepted": float(g.acc.mean()), "ci95": boot_ci(g.acc.values, g.pid.values)}
        for name, g in (("under_20_weeks", idx[idx.ga < 20]), ("20_weeks_or_more", idx[idx.ga >= 20])):
            r[name] = {"n": int(len(g)), "accepted": float(g.acc.mean()), "ci95": boot_ci(g.acc.values, g.pid.values)}
        res[var] = r
    json.dump(res, open(out / "hc18_age_bands.json", "w"), indent=2)
    print(json.dumps({v: {k: round(x["accepted"], 3) for k, x in r.items()} for v, r in res.items()}, indent=1))


if __name__ == "__main__":
    main()
