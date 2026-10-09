"""Stage 2 (HC18-augmented): one test evaluation of the retrained head check against the earlier FETAL-only check and
the CURRENT image check. Nothing is trained or tuned here.

Test sets: the same FETAL_PLANES test split as before (OLD/splits.csv), the 218 HC18 scans the current model never saw,
and the 335 unlabeled HC18 test_set scans. Intervals are 95%, bootstrapped over patients (FETAL_PLANES), linked-scan
groups (HC18 held-out) or single scans (test_set).

    python scripts/head_check_hc18_final.py --old eval_head_check --out eval_head_check_hc18
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from head_check_common import CACHE, load_fetal, rate
from head_check_extract import load_encoder
from head_check_final import auc_ci, groups_table, montage, score
from hcml.growth import hadlock_ga_weeks
from hcml.pipeline import Predictor
from hcml.preprocess import clean, read_gray, to_model_input
from hcml.validity import Reference

LABELLED = ("Cryptic", "Possibly cryptic")
BANDS = ["under 17 weeks", "17 to 20 weeks", "20 weeks or more"]


def band_of(ga):
    return pd.cut(pd.Series(ga), [0, 17, 20, 200], labels=BANDS, right=False)


def load_models(out_old: Path, out_new: Path):
    m = {}
    for v in ("A_with_other", "B_without_other"):
        m[("old", v)] = np.load(out_old / f"head_classifier_{v}.npz")
        m[("new", v)] = np.load(out_new / f"head_classifier2_{v}.npz")
    return m


def hc18_rows(emb, groups, ga_by_row, ref, models, extra_mask=None):
    """Acceptance for every method on one HC18 set, overall and by age band."""
    d = np.array([ref.distance(e) for e in emb])
    cur, cur_good = d <= ref.reject, d <= ref.warn
    methods = {"current_not_refused": cur, "current_good_only": cur_good}
    for (which, v), z in models.items():
        for tg in (95, 98):
            acc = score(z, emb) >= float(z[f"threshold_{tg}"])
            methods[f"{which}_{v[0]}_{tg}"] = acc
            methods[f"{which}_{v[0]}_{tg}_AND_current"] = acc & cur
    bands = band_of(ga_by_row) if ga_by_row is not None else None
    res = {"n": int(len(emb)), "overall": {k: rate(m, groups) for k, m in methods.items()}}
    if bands is not None:
        res["by_band"] = {}
        for b in BANDS:
            m = (bands == b).to_numpy()
            if m.any():
                res["by_band"][b] = {"n": int(m.sum()), **{k: rate(a[m], groups[m]) for k, a in methods.items()}}
    return res, methods, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="eval_head_check")
    ap.add_argument("--out", default="eval_head_check_hc18")
    args = ap.parse_args()
    old, out = Path(args.old), Path(args.out)

    models = load_models(old, out)
    ref = Reference.load("artifacts/ood_reference.npz")

    # ---------- FETAL_PLANES test split (identical to the earlier run) ----------
    df, emb = load_fetal()
    sp = pd.read_csv(old / "splits.csv")
    assert list(sp["image"]) == list(df["image"])
    te = (sp["split"] == "test").to_numpy()
    t, e = df[te].reset_index(drop=True), emb[te]
    head, pat = t["head"].to_numpy(), t["patient"].to_numpy()
    dist = np.array([ref.distance(x) for x in e])
    acc_cur, acc_good = dist <= ref.reject, dist <= ref.warn
    pipe = pd.read_csv(old / "test_pipeline.csv").set_index("image").loc[t["image"]].reset_index()
    ga_ok = pipe["ga_weeks"].notna().to_numpy()
    label_ok = (pipe["verdict_No"].isin(LABELLED) | pipe["verdict_Not sure"].isin(LABELLED)).to_numpy()
    verdict_ok = (pipe["verdict_No"] != "Cannot assess").to_numpy()

    report = {"fetal_test": {"images": int(len(t)), "patients": int(t["patient"].nunique()), "heads": int(head.sum()),
                             "thalamic": int(t["thalamic"].sum()), "always_not_head_baseline": float(1 - head.mean())},
              "variants": {}}
    per_image = t[["image", "path", "cls", "subplane", "machine", "patient", "head", "is_other"]].copy()
    per_image["current_distance"], per_image["current_not_refused"] = dist, acc_cur
    for v in ("A_with_other", "B_without_other"):
        s_old, s_new = score(models[("old", v)], e), score(models[("new", v)], e)
        per_image[f"score_old_{v[0]}"], per_image[f"score_new_{v[0]}"] = s_old, s_new
        from head_check_common import auc
        blk = {}
        blk["auc_old"] = {"value": auc(s_old[head], s_old[~head]), "ci95": auc_ci(s_old, head, pat)}
        blk["auc_new"] = {"value": auc(s_new[head], s_new[~head]), "ci95": auc_ci(s_new, head, pat)}
        blk["auc_current"] = {"value": auc(-dist[head], -dist[~head]), "ci95": auc_ci(-dist, head, pat)}
        for tg in (95, 98):
            a_old = s_old >= float(models[("old", v)][f"threshold_{tg}"])
            a_new = s_new >= float(models[("new", v)][f"threshold_{tg}"])
            tab_new = groups_table(t, a_new, acc_cur, acc_good)
            tab_old = groups_table(t, a_old, acc_cur, acc_good)
            tab = tab_new.rename(columns={"new": "new_hc18", "new_lo": "new_hc18_lo", "new_hi": "new_hc18_hi", "new_k": "new_hc18_k"})
            for c, nm in (("new", "old_fetal_only"), ("new_lo", "old_fetal_only_lo"), ("new_hi", "old_fetal_only_hi"), ("new_k", "old_fetal_only_k")):
                tab[nm] = tab_old[c].values
            comb = a_new & acc_cur
            sel = {"non-head, all": ~head, "non-head, without Other": ~head & ~t["is_other"].to_numpy()}
            for c in ("Fetal abdomen", "Fetal femur", "Fetal thorax", "Maternal cervix", "Other"):
                sel[c] = (t["cls"] == c).to_numpy()
            combo = {k: {"n": int(m.sum()),
                         "current_alone_gets_age": rate(ga_ok[m], pat[m]), "combined_gets_age": rate((ga_ok & comb)[m], pat[m]),
                         "current_alone_gets_cryptic_or_possibly": rate(label_ok[m], pat[m]),
                         "combined_gets_cryptic_or_possibly": rate((label_ok & comb)[m], pat[m])} for k, m in sel.items()}
            combo["heads"] = {"n": int(head.sum()),
                              "current_alone_gets_age": rate(ga_ok[head], pat[head]), "combined_gets_age": rate((ga_ok & comb)[head], pat[head]),
                              "current_alone_gets_a_verdict": rate(verdict_ok[head], pat[head]),
                              "combined_gets_a_verdict": rate((verdict_ok & comb)[head], pat[head])}
            blk[f"target_{tg}"] = {"threshold_new": float(models[("new", v)][f"threshold_{tg}"]),
                                   "threshold_old": float(models[("old", v)][f"threshold_{tg}"]),
                                   "groups": tab.round(4).to_dict("records"), "combined_with_current": combo,
                                   "decision_accuracy": {"new": float((a_new == head).mean()), "old": float((a_old == head).mean()),
                                                         "current": float((acc_cur == head).mean())}}
            if tg == 95:
                per_image[f"accept_new_{v[0]}"], per_image[f"accept_old_{v[0]}"] = a_new, a_old
        report["variants"][v] = blk
    for c in ("ga_weeks", "badge", "verdict_No", "verdict_Not sure"):
        per_image[c] = pipe[c].values

    # ---------- HC18 held-out 218 ----------
    part = pd.read_csv(out / "hc18_partition.csv")
    hidx = pd.read_csv(CACHE / "hc18_train_index.csv")
    assert list(part["filename"]) == list(hidx["filename"])
    hemb = np.load(CACHE / "hc18_train_emb.npy")
    ho = (part["part"] == "heldout").to_numpy()
    assert int(ho.sum()) == 218
    ga_h = hadlock_ga_weeks(hidx.loc[ho, "head circumference (mm)"].to_numpy())
    r_h, m_h, d_h = hc18_rows(hemb[ho], part.loc[ho, "pid"].to_numpy(), ga_h, ref,
                              {k: z for k, z in models.items()})
    r_h["age_basis"] = "Hadlock age from the scan's head circumference"
    # ---------- HC18 unlabeled test_set 335 (no HC: band by the app model's own estimated age) ----------
    tidx = pd.read_csv(CACHE / "hc18_testset_index.csv")
    temb = np.load(CACHE / "hc18_testset_emb.npy")
    pred = Predictor("artifacts")
    est = []
    for p in tidx["path"]:
        r = pred.predict(Path(p).read_bytes(), Path(p).name, 1.0)
        est.append(np.nan if r.ga_weeks is None else r.ga_weeks)
    est = np.array(est)
    r_t, m_t, d_t = hc18_rows(temb, np.arange(len(temb)), np.where(np.isnan(est), -1, est), ref, models)
    r_t["age_basis"] = ("the app model's ESTIMATED age (no head circumference is published for test_set); "
                        f"{int(np.isnan(est).sum())} scans refused by the current check have no estimate and are in no band")
    report["hc18_heldout_218"], report["hc18_testset_335"] = r_h, r_t
    pd.DataFrame({"image": tidx["image"], "est_age_weeks": est, "current_dist": d_t,
                  **{k: v for k, v in m_t.items() if k.endswith("_95") or k.endswith("_98")}}).to_csv(out / "hc18_testset_scores.csv", index=False)

    # ---------- the repo's own fixtures (HC18-style heads and the two first-trimester images) ----------
    enc, den = load_encoder("artifacts")
    fx = sorted(Path("tests/fixtures").glob("*.png"))
    fe = enc.predict(np.stack([to_model_input(clean(read_gray(p.read_bytes(), p.name), denoise=den)) for p in fx]), verbose=0).mean(axis=(1, 2))
    fixtures = []
    for p, x in zip(fx, fe):
        row = {"file": p.name, "current_distance": float(ref.distance(x)), "current_refused": bool(ref.distance(x) > ref.reject)}
        for (which, v), z in models.items():
            row[f"{which}_{v[0]}_score"] = float(score(z, x[None])[0])
            row[f"{which}_{v[0]}_accepts_at_95"] = bool(score(z, x[None])[0] >= float(z["threshold_95"]))
        fixtures.append(row)
    report["fixtures"] = fixtures
    json.dump(report, open(out / "test_report.json", "w"), indent=2)
    per_image.drop(columns=["path"]).to_csv(out / "fetal_test_images.csv", index=False)

    # ---------- examples of mistakes, both directions (variant A, 95%) ----------
    thr = float(models[("new", "A_with_other")]["threshold_95"])
    s = per_image["score_new_A"].to_numpy()
    so = per_image["score_old_A"].to_numpy()

    def lines(r, thr=thr):
        return [f"{r['cls']} / {r['subplane'] if r['head'] else '-'} | {r['machine']}",
                f"new {r['score_new_A']:+.2f} (thr {thr:+.2f}) | old {r['score_old_A']:+.2f}",
                f"current d={r['current_distance']:.3f} {'accepted' if r['current_not_refused'] else 'REFUSED'}"]
    fa = per_image[(~per_image["head"]) & (s >= thr)].sort_values("score_new_A", ascending=False)
    fa = fa.groupby("cls", group_keys=False).head(3).head(12)
    montage([{**r, "lines": lines(r)} for _, r in fa.iterrows()], out / "examples_new_false_accepts.png",
            "RETRAINED check: non-head images it wrongly accepts (highest scores)")
    fr = per_image[per_image["head"] & (s < thr)]
    fr = pd.concat([fr[fr["subplane"] == "Trans-thalamic"].sort_values("score_new_A").head(8),
                    fr[fr["subplane"] != "Trans-thalamic"].sort_values("score_new_A").head(4)])
    montage([{**r, "lines": lines(r)} for _, r in fr.iterrows()], out / "examples_new_false_refusals_fetal.png",
            "RETRAINED check: FETAL_PLANES head images it wrongly refuses (lowest scores)")
    zA = models[("new", "A_with_other")]
    zO = models[("old", "A_with_other")]
    rows = []
    for name, idx_df, e_, dd in (("HC18 held-out", hidx[ho].reset_index(drop=True), hemb[ho], d_h),
                                 ("HC18 test_set", tidx, temb, d_t)):
        sc, so_ = score(zA, e_), score(zO, e_)
        for i in np.flatnonzero(sc < thr):
            rows.append({"path": idx_df["path"].iloc[i], "set": name, "score": sc[i], "old": so_[i], "d": dd[i]})
    rf = pd.DataFrame(rows).sort_values("score")
    if len(rf):
        pick = rf.iloc[np.linspace(0, len(rf) - 1, min(12, len(rf))).astype(int)]
        montage([{"path": r["path"], "lines": [f"{r['set']} head scan", f"new {r['score']:+.2f} (thr {thr:+.2f}) REFUSED | old {r['old']:+.2f}",
                                              f"current d={r['d']:.3f} {'accepted' if r['d'] <= ref.reject else 'REFUSED'}"]}
                 for _, r in pick.iterrows()], out / "examples_new_false_refusals_hc18.png",
                "RETRAINED check: HC18 head scans it still refuses (spread across scores)")
    print("refused HC18 heads (held-out + test_set), retrained A@95:", len(rf), "of", 218 + 335)
    print("done ->", out)


if __name__ == "__main__":
    main()
