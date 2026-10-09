"""Confirm that no training or validation image is in the testing set of project_package/.

    python scripts/package_check_no_leak.py --pkg project_package --fetal DIR --hc18 DIR

Checks (all must pass):
  1. the image files in 02_testing_set/images match the three test CSVs exactly (2,478 / 218 / 335);
  2. no test file name of FETAL_PLANES_DB or of the HC18 held-out set is in the training/validation lists;
  3. no FETAL_PLANES patient and no HC18 linked-scan group is on both sides;
  4. no test image has the same decoded pixels (SHA-256 of the pixel array) as any training or validation image,
     read from the original datasets, so renamed or re-encoded copies are caught too;
  5. the same for the exact file bytes.
The HC18 test_set re-uses file names of the HC18 training set (e.g. 000_HC.png exists in both releases), so for it the
name overlap is reported but the pixel comparison decides. Writes OUT/leak_check.json and prints a short report.
"""
import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def pixel_hash(path):
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest(), hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes(paths):
    with Pool(4) as p:
        return p.map(pixel_hash, [str(x) for x in paths], chunksize=32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", default="project_package")
    ap.add_argument("--fetal", required=True, help="unzipped FETAL_PLANES_ZENODO folder (original training images)")
    ap.add_argument("--hc18", required=True, help="HC18 folder with training_set/training_set")
    a = ap.parse_args()
    pkg = Path(a.pkg)
    t02, t01 = pkg / "02_testing_set", pkg / "01_training_set"

    tr = pd.read_csv(t01 / "train.csv")
    va = pd.read_csv(t01 / "validation.csv")
    trv = pd.concat([tr, va])
    tests = {"FETAL_PLANES_DB": ("barcelona_fetal_planes_test", pd.read_csv(t02 / "barcelona_fetal_planes_test.csv")),
             "HC18_heldout": ("hc18_heldout_218", pd.read_csv(t02 / "hc18_heldout_218.csv")),
             "HC18_test_set": ("hc18_test_set_335", pd.read_csv(t02 / "hc18_test_set_335.csv"))}
    expect = {"FETAL_PLANES_DB": 2478, "HC18_heldout": 218, "HC18_test_set": 335}
    res, ok = {}, True

    # 1. folder contents == CSV rows
    for k, (folder, df) in tests.items():
        files = sorted(p.name for p in (t02 / "images" / folder).glob("*.png"))
        match = files == sorted(df["file_name"]) and len(files) == expect[k]
        res[f"{k}: files in folder == CSV rows == {expect[k]}"] = bool(match)
        ok &= match

    # 2. names
    tr_names = {d: set(trv.loc[trv["dataset"] == d, "file_name"]) for d in ("FETAL_PLANES_DB", "HC18")}
    n_f = len(set(tests["FETAL_PLANES_DB"][1]["file_name"]) & tr_names["FETAL_PLANES_DB"])
    n_h = len(set(tests["HC18_heldout"][1]["file_name"]) & tr_names["HC18"])
    n_t = len(set(tests["HC18_test_set"][1]["file_name"]) & tr_names["HC18"])
    res["FETAL_PLANES test file names also in training/validation"] = n_f
    res["HC18 held-out file names also in training/validation"] = n_h
    res["HC18 test_set file names that equal an HC18 training/validation name (name re-use between releases; pixels decide)"] = n_t
    ok &= (n_f == 0 and n_h == 0)

    # 3. patients / linked-scan groups
    pf = set(tests["FETAL_PLANES_DB"][1]["patient_id"]) & set(trv.loc[trv["dataset"] == "FETAL_PLANES_DB", "patient_or_group_id"].astype(int))
    gh = set(tests["HC18_heldout"][1]["group_id"]) & set(trv.loc[trv["dataset"] == "HC18", "patient_or_group_id"])
    res["FETAL_PLANES patients on both sides"] = len(pf)
    res["HC18 linked-scan groups on both sides (held-out vs training/validation)"] = len(gh)
    ok &= (len(pf) == 0 and len(gh) == 0)

    # 4./5. pixels and bytes
    fe, hc = Path(a.fetal), Path(a.hc18)
    train_paths = [fe / "Images" / f"{n}.png" for n in trv.loc[trv["dataset"] == "FETAL_PLANES_DB", "file_name"].str.replace(".png", "", regex=False)]
    train_paths += [hc / "training_set" / "training_set" / n for n in trv.loc[trv["dataset"] == "HC18", "file_name"]]
    th = hashes(train_paths)
    train_pix, train_byt = {h[0] for h in th}, {h[1] for h in th}
    res["training+validation images hashed"] = len(train_paths)
    total_pix = total_byt = 0
    for k, (folder, df) in tests.items():
        h = hashes([t02 / "images" / folder / n for n in df["file_name"]])
        p_hit = sum(x[0] in train_pix for x in h)
        b_hit = sum(x[1] in train_byt for x in h)
        res[f"{k}: test images with the same pixels as a training/validation image"] = p_hit
        res[f"{k}: test images with the same file bytes as a training/validation image"] = b_hit
        total_pix += p_hit
        total_byt += b_hit
    ok &= (total_pix == 0 and total_byt == 0)
    res["RESULT"] = "PASS: no training or validation image is in the testing set" if ok else "FAIL"
    (t02 / "leak_check.json").write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print(f"{k}: {v}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
