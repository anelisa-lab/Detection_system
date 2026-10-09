"""Stress-test the committed model on FETAL_PLANES_DB (Zenodo 3904280): many ultrasound planes, not just HC18.

Nothing is trained and nothing in artifacts/ is changed. Every image goes through the same code path the app
uses for an upload: `Predictor.predict(bytes, name)` (image check, age estimate, Grad-CAM skull check) and then
`screen(...)` (the Cryptic / Possibly cryptic / Not cryptic / Cannot assess rule).

    python scripts/eval_fetal_planes.py --data <unzipped FETAL_PLANES_ZENODO dir> --out eval_fetal_planes

The dataset has no gestational age, so age estimates can only be checked for plausibility.
Dataset: Burgos-Artizzu et al., Sci Rep 2020, doi:10.1038/s41598-020-67076-5; CC BY 4.0.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd

from hcml.model import overlay
from hcml.pipeline import Predictor
from hcml.screening import ANSWERS, CANNOT, CRYPTIC, NO, NOT_CRYPTIC, NOT_SURE, POSSIBLY, YES, screen
from hcml.skull import ellipse_mask, peak_inside

HEAD_CLASS = "Fetal brain"
AGE_LO, AGE_HI = 12.0, 40.0
ON_SKULL_MIN = 0.5


def find_csv(root: Path) -> Path:
    hits = sorted(root.rglob("*.csv"))
    if not hits:
        raise FileNotFoundError(f"no CSV under {root}")
    return hits[0]


def load_table(root: Path) -> pd.DataFrame:
    csv = find_csv(root)
    df = pd.read_csv(csv, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    img_dir = next((p for p in root.rglob("Images") if p.is_dir()), root)
    df["path"] = df["Image_name"].map(lambda n: str(img_dir / f"{n}.png") if (img_dir / f"{n}.png").exists()
                                      else str(img_dir / f"{n}.jpg"))
    df["class"] = df["Plane"].str.strip()
    df["subplane"] = df["Brain_plane"].astype(str).str.strip()
    return df[df["path"].map(lambda p: Path(p).exists())].reset_index(drop=True)


def grouped_sample(df: pd.DataFrame, per_class: int, max_per_patient: int, seed: int) -> pd.DataFrame:
    """Up to `per_class` images per plane (brain: per sub-plane), at most `max_per_patient` from any one patient."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby(["class", "subplane"]):
        g = g.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        g = g.groupby("Patient_num", group_keys=False).head(max_per_patient)
        parts.append(g.head(per_class))
    return pd.concat(parts).reset_index(drop=True)


def run_images(sample: pd.DataFrame, pred: Predictor) -> pd.DataFrame:
    rows = []
    for k, (_, r) in enumerate(sample.iterrows()):
        data = Path(r["path"]).read_bytes()
        try:
            res = pred.predict(data, Path(r["path"]).name, 1.0)
        except Exception as e:                     # an unreadable image is reported, not hidden
            rows.append({"image": r["Image_name"], "error": str(e)})
            continue
        peak = None
        if res.cam is not None and res.ellipse is not None:
            peak = bool(peak_inside(res.cam, ellipse_mask(res.ellipse)))
        row = {
            "image": r["Image_name"], "path": r["path"], "class": r["class"], "subplane": r["subplane"], "patient": r["Patient_num"],
            "machine": r["US_Machine"], "distance": res.distance, "level": res.level, "badge": res.badge,
            "reasons": "; ".join(res.badge_reasons), "overlap": res.overlap, "peak_on_skull": peak,
            "skull_found": res.ellipse is not None, "ga_weeks": res.ga_weeks, "half_days": res.half_days,
        }
        for ans in ANSWERS:                         # the app's rule, for every answer, from the same Result
            s = screen(res.ga_weeks, res.half_days, res.badge, ans, None, "; ".join(res.badge_reasons))
            row[f"verdict_{ans}"] = s.badge
            row[f"label_{ans}"] = s.label
        rows.append(row)
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(sample)}", flush=True)
    return pd.DataFrame(rows)


def boot_ci(values, patients, stat, n_boot=2000, seed=0):
    """95% CI of stat(values) resampling whole patients (clusters)."""
    values, patients = np.asarray(values), np.asarray(patients)
    if len(values) == 0:
        return [None, None]
    groups = [np.flatnonzero(patients == p) for p in np.unique(patients)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        v = stat(values[idx])
        if v == v:
            out.append(v)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))] if out else [None, None]


def prop(mask, patients):
    mask = np.asarray(mask, dtype=float)
    return {"value": float(mask.mean()) if len(mask) else None, "ci95": boot_ci(mask, patients, np.mean)}


def summarise(df: pd.DataFrame) -> dict:
    out = {}
    for cls, g in df.groupby("class"):
        p = g["patient"].to_numpy()
        acc = g[g["level"] != "rejected"]
        pa = acc["patient"].to_numpy()
        item = {
            "n_images": int(len(g)), "n_patients": int(g["patient"].nunique()),
            "level": {k: prop(g["level"] == k, p) for k in ("good", "reduced", "rejected")},
            "badge": {k: prop(g["badge"] == k, p) for k in ("Good", "Limited", "Poor", "Rejected")},
            "distance_median": float(g["distance"].median()),
            "distance_ge_p90_0.182": prop(g["distance"] > 0.1822, p),
        }
        if len(acc):
            item["n_accepted"] = int(len(acc))
            item["on_skull_overlap_median"] = {"value": float(acc["overlap"].median()),
                                               "ci95": boot_ci(acc["overlap"].to_numpy(), pa, np.median)}
            item["on_skull_below_50pct"] = prop(acc["overlap"].to_numpy() < ON_SKULL_MIN, pa)
            item["no_skull_found"] = prop(~acc["skull_found"].to_numpy(), pa)
            ga = acc["ga_weeks"].to_numpy()
            item["age_weeks"] = {"min": float(ga.min()), "p5": float(np.percentile(ga, 5)),
                                 "median": float(np.median(ga)), "p95": float(np.percentile(ga, 95)),
                                 "max": float(ga.max())}
            item["age_outside_12_40"] = prop((ga < AGE_LO) | (ga > AGE_HI), pa)
            item["age_ge_20"] = prop(ga >= 20, pa)
        out[cls] = item
    return out


def expected_rule(ga, half, image_badge, answer):
    """The rule as stated for this project, written independently of hcml/screening.py.

    Returns (expected verdict badge, range straddles 20 weeks, should be softened to "May be ...")."""
    if image_badge in ("Poor", "Rejected") or ga is None or ga != ga:
        return CANNOT, False, False
    low, high = ga - half / 7, ga + half / 7
    soft = image_badge == "Limited"
    if low < 20 <= high:                               # both readings are shown; the answer decides the badge
        return (NOT_CRYPTIC if answer == YES else POSSIBLY), True, soft
    if ga < 20:
        return NOT_CRYPTIC, False, soft
    return {NO: CRYPTIC, YES: NOT_CRYPTIC, NOT_SURE: POSSIBLY}[answer], False, soft


def label_checks(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Step 5: 20 images x three answers, checked against an independent statement of the rule.

    Mix: 7 clearly under 20 weeks, 7 clearly 20 or more, 3 whose range straddles 20, 3 Poor/Rejected."""
    rng = np.random.default_rng(seed)
    ok = df[df["badge"].isin(["Good", "Limited"])].copy()
    ok["low"], ok["high"] = ok["ga_weeks"] - ok["half_days"] / 7, ok["ga_weeks"] + ok["half_days"] / 7
    groups = [(ok[ok["high"] < 20], 7), (ok[ok["low"] >= 20], 7),
              (ok[(ok["low"] < 20) & (ok["high"] >= 20)], 3), (df[df["badge"].isin(["Poor", "Rejected"])], 3)]
    pick = pd.concat([g.sample(min(len(g), n), random_state=int(rng.integers(1 << 31))) for g, n in groups])
    rows = []
    for _, r in pick.iterrows():
        ga, half = r["ga_weeks"], r["half_days"]
        for ans in ANSWERS:
            s = screen(None if r["badge"] == "Rejected" else ga, half, r["badge"], ans, None, r["reasons"])
            want, straddles, soft = expected_rule(ga, half, r["badge"], ans)
            if want == CANNOT:
                ok_rule = s.badge == CANNOT and "see a clinician regardless" in s.sentence
            elif straddles:
                ok_rule = s.borderline and len(s.readings) == 2 and s.badge == want
            else:
                ok_rule = (not s.borderline) and s.badge == want
            soft_ok = True if want == CANNOT else (
                (s.label.startswith("May be") or (s.borderline and s.readings[0][1].startswith("May be"))) == soft)
            rows.append({"image": r["image"], "class": r["class"], "ga_weeks": None if r["badge"] == "Rejected" else round(ga, 1),
                         "range_days": round(half, 1), "image_check": r["badge"], "answer": ans, "label": s.label,
                         "verdict": s.badge, "expected_verdict": want, "rule_ok": bool(ok_rule), "softened_ok": bool(soft_ok)})
    return pd.DataFrame(rows)


def montage(items, pred, root, path, title):
    """Grad-CAM overlays with the fitted skull ellipse, one tile per image, with its scores printed on it."""
    tiles = []
    for row in items:
        res = pred.predict(Path(row["path"]).read_bytes(), Path(row["path"]).name, 1.0)
        tile = overlay(res.gray, res.cam) if res.cam is not None else cv2.cvtColor(
            cv2.cvtColor(res.gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        tile = cv2.resize(tile, (320, 320), interpolation=cv2.INTER_LINEAR)
        if res.ellipse is not None:
            (cx, cy), (a, b), ang = res.ellipse
            k = 320 / res.gray.shape[0]
            cv2.ellipse(tile, ((cx * k, cy * k), (a * k, b * k), ang), (255, 255, 255), 2, cv2.LINE_AA)
        ga = "no age" if row["ga_weeks"] != row["ga_weeks"] or row["ga_weeks"] is None else f"{row['ga_weeks']:.1f} wk"
        lines = [f"{row['class']} {row['subplane'] if row['subplane'] not in ('nan', 'Not A Brain') else ''}".strip(),
                 f"d={row['distance']:.3f} {row['level']} | {row['badge']}", f"{ga} | {row['verdict_No']}"]
        pad = np.full((64, 320, 3), 255, np.uint8)
        for j, t in enumerate(lines):
            cv2.putText(pad, t[:44], (4, 16 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([tile, pad]))
    if not tiles:
        return
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.putText(grid, title, (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="unzipped FETAL_PLANES_ZENODO folder")
    ap.add_argument("--out", default="eval_fetal_planes", help="output folder (never artifacts/)")
    ap.add_argument("--model-dir", default="artifacts")
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--max-per-patient", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--from-csv", action="store_true", help="reuse OUT/per_image.csv instead of re-running the model")
    args = ap.parse_args()
    out = Path(args.out)
    if out.resolve() == Path(args.model_dir).resolve():
        sys.exit("--out must not be the model folder")
    (out / "examples").mkdir(parents=True, exist_ok=True)

    pred = Predictor(args.model_dir)
    if args.from_csv:
        df = pd.read_csv(out / "per_image.csv")
    else:
        table = load_table(Path(args.data))
        print("images in dataset:", len(table), table["class"].value_counts().to_dict())
        sample = grouped_sample(table, args.per_class, args.max_per_patient, args.seed)
        print("sample:", sample["class"].value_counts().to_dict(), "patients:", sample["Patient_num"].nunique())
        df = run_images(sample, pred)
        df.to_csv(out / "per_image.csv", index=False)
    if "error" in df:
        print("unreadable images:", int(df["error"].notna().sum()))
        df = df[df["error"].isna()]

    summary = summarise(df)
    nh = df[df["class"] != HEAD_CLASS]
    fails = {ans: nh[nh["ga_weeks"].notna() & nh[f"verdict_{ans}"].isin([CRYPTIC, POSSIBLY])] for ans in (NO, NOT_SURE)}
    safety = {
        "non_head_images": int(len(nh)),
        "non_head_with_an_age": int(nh["ga_weeks"].notna().sum()),
        "non_head_with_age_and_cryptic_or_possibly_when_answer_No": int(len(fails[NO])),
        "non_head_with_age_and_cryptic_or_possibly_when_answer_NotSure": int(len(fails[NOT_SURE])),
        "by_class": {c: {"n": int((nh["class"] == c).sum()),
                         "with_age": int(((nh["class"] == c) & nh["ga_weeks"].notna()).sum()),
                         "cryptic_or_possibly_when_No": int((fails[NO]["class"] == c).sum()),
                         "cryptic_or_possibly_when_NotSure": int((fails[NOT_SURE]["class"] == c).sum())}
                     for c in sorted(nh["class"].unique())},
    }
    union = pd.concat([fails[NO], fails[NOT_SURE]]).drop_duplicates("image")
    union.to_csv(out / "safety_failures.csv", index=False)

    checks = label_checks(df, args.seed)
    checks.to_csv(out / "label_checks.csv", index=False)
    json.dump({"images_evaluated": int(len(df)), "n_patients": int(df["patient"].nunique()),
               "per_class": summary, "safety": safety,
               "label_check": {"rows": int(len(checks)), "images": int(checks["image"].nunique()),
                               "rule_ok": int(checks["rule_ok"].sum()), "softened_ok": int(checks["softened_ok"].sum())}},
              open(out / "summary.json", "w"), indent=2)

    # a spread of failures across classes (closest to the training scans first), and the brain images the check refused
    pick = (union.sort_values("distance").groupby("class", group_keys=False).head(3).head(12) if len(union) else union)
    montage(pick.to_dict("records"), pred, None, out / "examples" / "non_head_accepted.png",
            "non-head images that got an age and a Cryptic/Possibly label (white ring = fitted skull)")
    refused = df[(df["class"] == HEAD_CLASS) & (df["level"] == "rejected")]
    if len(refused):
        thal = refused[refused["subplane"] == "Trans-thalamic"].sort_values("distance", ascending=False)
        rest = refused[refused["subplane"] != "Trans-thalamic"].sort_values("distance", ascending=False)
        montage(pd.concat([thal.head(8), rest.head(4)]).to_dict("records"), pred, None,
                out / "examples" / "brain_refused.png", "brain images refused by the input check (thalamic first)")
    print("done ->", out)


if __name__ == "__main__":
    main()
