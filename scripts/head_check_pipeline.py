"""Stage 2: run the CURRENT app pipeline (image check, age, Grad-CAM check, screening rule) on the test-split images.

The current pipeline is fixed and does not depend on the new classifier, so this can run before the final comparison.
Needs OUT/splits.csv from head_check_fit.py. Reuses the loop from eval_fetal_planes.py (same code path as the app).

    python scripts/head_check_pipeline.py --out eval_head_check
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from eval_fetal_planes import run_images
from head_check_common import CACHE
from hcml.pipeline import Predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_head_check")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    out = Path(args.out)
    if not (CACHE / "fetal_list.csv").exists():
        raise SystemExit(f"{CACHE / 'fetal_list.csv'} not found: run scripts/head_check_make_lists.py first, or set HEAD_CHECK_CACHE")
    lst = pd.read_csv(CACHE / "fetal_list.csv")
    sp = pd.read_csv(out / "splits.csv")
    assert list(lst["image"]) == list(sp["image"])
    lst["split"] = sp["split"].values
    rows = lst[lst["split"] == args.split].copy()
    rows["Image_name"], rows["class"] = rows["image"], rows["Plane"].str.strip()
    rows["subplane"] = rows["Brain_plane"].astype(str).str.strip()
    print(f"{len(rows)} {args.split} images")
    df = run_images(rows, Predictor("artifacts"))
    df.to_csv(out / f"{args.split}_pipeline.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
