"""Export the retrained head check from the study folder to its own files under artifacts/head_check/.

Writes classifier.npz and thresholds.json only. It never touches model.keras, metadata.json, ood_reference.npz or
locator.keras. Re-run only if the study is repeated.

    python scripts/export_head_check.py --study eval_head_check_hc18 --out artifacts/head_check
"""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

VARIANT = "A_with_other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="eval_head_check_hc18")
    ap.add_argument("--out", default="artifacts/head_check")
    args = ap.parse_args()
    study, out = Path(args.study), Path(args.out)
    if out.name != "head_check":
        raise SystemExit("--out must be a folder named head_check (so the existing artifacts are never overwritten)")
    z = np.load(study / f"head_classifier2_{VARIANT}.npz")
    val = json.load(open(study / "val_report.json"))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "classifier.npz", mean=z["mean"], scale=z["scale"], coef=z["coef"],
                        intercept=z["intercept"])
    meta = {
        "default": 95,
        "operating_points": {k: float(z[f"threshold_{k}"]) for k in ("90", "95", "98")},
        "provenance": {
            "what": "logistic regression (class-balanced) on the frozen encoder's 2048-d embedding; head vs not head",
            "variant": "A: 'Other' plane used as a negative class",
            "C": float(z["C"]),
            "trained_on": "FETAL_PLANES_DB train patients (brain planes = head) + HC18 head scans from the current "
                          "model's own train/validation pool (613 train); the 218 held-out HC18 scans and the 335 "
                          "unlabeled HC18 test_set scans were never used",
            "threshold_rule": "highest score that still accepts the stated share of ALL validation heads "
                              "(FETAL_PLANES + HC18 validation scans pooled)",
            "validation": val["variants"][VARIANT]["at_target"],
            "exported": date.today().isoformat(),
            "study": "scripts/head_check_hc18_fit.py, scripts/head_check_hc18_final.py, eval_head_check_hc18/",
            "data": "FETAL_PLANES_DB, Burgos-Artizzu et al., Sci Rep 10:10200 (2020), CC BY 4.0, "
                    "doi:10.5281/zenodo.3904280; HC18, van den Heuvel et al., PLoS ONE 13(8):e0200412 (2018), CC BY 4.0",
        },
    }
    json.dump(meta, open(out / "thresholds.json", "w"), indent=2)
    print("wrote", out / "classifier.npz", "and", out / "thresholds.json", meta["operating_points"])


if __name__ == "__main__":
    main()
