"""Run named cases through the real Streamlit app (AppTest) and record what the page shows, for before/after checks.

    python scripts/wiring_cases.py --root . --label after --cases wiring_cases.json --out eval_wiring
    python scripts/wiring_cases.py --root /path/to/older/checkout --label before ...

`--root` is the checkout whose app.py is exercised (so an older commit can be checked out next to this one). The cases
file maps a name to {"path": ..., "answers": ["Yes", "No", "Not sure"]}. Set HEAD_CHECK_TARGET=98 to try the other
operating point. For each case and answer it records the age headline the page shows (or "No estimate"), the image
check badge, the screening badge and label, and the reason text on the No estimate card.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--label", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", default="eval_wiring")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "tests"))
    import test_app_ui as T

    cases = json.loads(Path(args.cases).read_text())
    rows = []
    for name, c in cases.items():
        for ans in c.get("answers", ["Not sure"]):
            at = T.run(Path(c["path"]), ans)
            h = T.html_of(at)
            hero = re.search(r'class="hero num"[^>]*>([^<]+)<', h)
            verdict = T.verdict_text(at) or (None, None)
            noest = re.search(r'class="hero muted">No estimate</div><div class="range">([^<]+)<', h)
            tile = re.search(r'Head-view check</div>\s*<div class="tile-v[^"]*">([^<]+)<', h)
            rows.append({"case": name, "file": Path(c["path"]).name, "answer": ans,
                         "age_headline": hero.group(1) if hero else "No estimate",
                         "image_check": T.image_badge(at), "screening_badge": verdict[0], "screening_label": verdict[1],
                         "no_estimate_reason": noest.group(1) if noest else "",
                         "head_view_tile_shown": bool(tile) or "Head-view check" in h})
            print(rows[-1], flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / f"cases_{args.label}.csv", index=False)


if __name__ == "__main__":
    main()
