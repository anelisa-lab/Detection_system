"""Step 0 of the head-check study: build the image lists that head_check_extract.py reads.

    python scripts/head_check_make_lists.py --fetal /path/to/unzipped/FETAL_PLANES_ZENODO \\
        --hc18 /path/to/hc18 --cache /home/user/head_check_cache

--fetal : the unzipped FETAL_PLANES_ZENODO folder (holds FETAL_PLANES_DB_data.csv and Images/)
--hc18  : a folder holding training_set/training_set/*.png, test_set/test_set/*.png and
          training_set_pixel_size_and_HC.csv (the layout produced by unzipping the two Zenodo zips as in README.txt)
The other study scripts expect the cache at /home/user/head_check_cache; symlink it if you use another place.
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetal", required=True)
    ap.add_argument("--hc18", required=True)
    ap.add_argument("--cache", default="/home/user/head_check_cache")
    a = ap.parse_args()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)

    fe = Path(a.fetal)
    df = pd.read_csv(fe / "FETAL_PLANES_DB_data.csv", sep=";")
    df.columns = [c.strip() for c in df.columns]
    df["image"] = df["Image_name"]
    df["path"] = str(fe / "Images") + "/" + df["Image_name"] + ".png"
    df.to_csv(cache / "fetal_list.csv", index=False)

    h = Path(a.hc18)
    tr = pd.read_csv(h / "training_set_pixel_size_and_HC.csv")
    tr.columns = [c.strip() for c in tr.columns]
    tr["image"] = tr["filename"]
    tr["path"] = tr["filename"].map(lambda f: str(h / "training_set" / "training_set" / f))
    assert all(Path(p).exists() for p in tr["path"]), "HC18 training images not found"
    tr.to_csv(cache / "hc18_train_list.csv", index=False)

    ts = sorted((h / "test_set" / "test_set").glob("*.png"))
    pd.DataFrame({"image": [p.name for p in ts], "path": [str(p) for p in ts]}).to_csv(cache / "hc18_testset_list.csv", index=False)
    print(f"fetal {len(df)} | hc18 train {len(tr)} | hc18 test_set {len(ts)} -> {cache}")


if __name__ == "__main__":
    main()
