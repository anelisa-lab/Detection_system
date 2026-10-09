"""Rebuild the head-check training and validation sets from the CSV lists in project_package/01_training_set.

The lists name images only; the images come from the two public datasets (download steps in 01_training_set/README.txt).

    python scripts/rebuild_training_set.py --lists project_package/01_training_set \\
        --fetal /path/to/FETAL_PLANES_ZENODO --hc18 /path/to/hc18 --out training_set_rebuilt [--link]

--fetal : unzipped FETAL_PLANES_ZENODO folder (has Images/)
--hc18  : folder with training_set/training_set/*.png (unzipped HC18 training_set.zip)
--link  : symlink instead of copy (instant, no extra disk)
Output: OUT/{train,validation}/{head,not_head}/<dataset>__<file name>.png and OUT/rebuilt_manifest.csv.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lists", required=True)
    ap.add_argument("--fetal", required=True)
    ap.add_argument("--hc18", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--link", action="store_true")
    a = ap.parse_args()
    lists, out = Path(a.lists), Path(a.out)
    rows = []
    for split, name in (("train", "train.csv"), ("validation", "validation.csv")):
        df = pd.read_csv(lists / name)
        for r in df.itertuples():
            if r.dataset == "FETAL_PLANES_DB":
                src = Path(a.fetal) / "Images" / r.file_name
            else:
                src = Path(a.hc18) / "training_set" / "training_set" / r.file_name
            if not src.exists():
                raise SystemExit(f"missing source image: {src}")
            dst = out / split / r.label.replace(" ", "_") / f"{r.dataset}__{r.file_name}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve()) if a.link else shutil.copy2(src, dst)
            rows.append({"split": split, "dataset": r.dataset, "label": r.label, "file": str(dst.relative_to(out))})
    m = pd.DataFrame(rows)
    m.to_csv(out / "rebuilt_manifest.csv", index=False)
    print(m.groupby(["split", "dataset", "label"]).size().to_string())
    print("total", len(m))


if __name__ == "__main__":
    main()
