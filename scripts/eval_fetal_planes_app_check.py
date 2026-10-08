"""Send a few FETAL_PLANES images through the real Streamlit app (AppTest) with each answer and compare the
screening badge the page shows with the project's rule. Run after eval_fetal_planes.py (needs its per_image.csv).

    python scripts/eval_fetal_planes_app_check.py --out eval_fetal_planes
"""
import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import pandas as pd

from eval_fetal_planes import expected_rule          # noqa: E402
from hcml.screening import ANSWERS                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_fetal_planes")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    df = pd.read_csv(out / "per_image.csv")
    df = df[df["error"].isna()] if "error" in df else df
    ok = df[df["badge"].isin(["Good", "Limited"])].copy()
    ok["low"], ok["high"] = ok["ga_weeks"] - ok["half_days"] / 7, ok["ga_weeks"] + ok["half_days"] / 7
    early, late = ok[ok["high"] < 20], ok[ok["low"] >= 20]
    mid = ok[(ok["low"] < 20) & (ok["high"] >= 20)]
    groups = [("clearly under 20 (Good)", early[early["badge"] == "Good"], 1), ("clearly under 20 (Limited)", early[early["badge"] == "Limited"], 1),
              ("clearly 20 or more (Good)", late[late["badge"] == "Good"], 1), ("clearly 20 or more (Limited)", late[late["badge"] == "Limited"], 1),
              ("range straddles 20", mid, 1), ("Poor", df[df["badge"] == "Poor"], 1), ("Rejected", df[df["badge"] == "Rejected"], 1)]
    pick = [(why, r) for why, g, n in groups for _, r in g.sample(min(len(g), n), random_state=args.seed).iterrows()]

    import test_app_ui as T                               # the project's own AppTest helpers
    rows = []
    for why, r in pick:
        for ans in ANSWERS:
            at = T.run(Path(r["path"]), ans)
            shown = T.verdict_text(at)                    # (badge text, headline) as rendered on the page
            want, straddles, soft = expected_rule(r["ga_weeks"], r["half_days"], r["badge"], ans)
            badge = shown[0] if shown else None
            rows.append({"image": r["image"], "class": r["class"], "why_chosen": why, "image_check_in_app": T.image_badge(at),
                         "answer": ans, "page_badge": badge, "page_headline": shown[1] if shown else None,
                         "expected": want, "match": badge == want,
                         "page_shows_age": bool(re.search(r'class="hero num"[^>]*>\s*\d+ weeks', T.html_of(at)))})
    res = pd.DataFrame(rows)
    res.to_csv(out / "app_check.csv", index=False)
    print(res[["why_chosen", "image_check_in_app", "answer", "page_badge", "expected", "match", "page_shows_age"]].to_string())
    print(f"\n{int(res['match'].sum())}/{len(res)} page results match the rule")


if __name__ == "__main__":
    main()
