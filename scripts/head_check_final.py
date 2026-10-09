"""Stage 3: the one test-set evaluation of the new head-view check against the CURRENT image check.

Reads the classifiers and thresholds frozen by head_check_fit.py (chosen on validation only), the current-pipeline
results from head_check_pipeline.py, and the embeddings. Nothing is trained or tuned here. All intervals are 95%,
bootstrapped over patients (HC18: over linked-scan groups).

    python scripts/head_check_final.py --out eval_head_check
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd

from head_check_common import CACHE, PRIMARY_TARGET, RECALL_TARGETS, auc, boot_ci, load_fetal, rate
from hcml.validity import Reference

VARIANTS = {"A_with_other": "A (trained with 'Other' as a negative)", "B_without_other": "B (trained without 'Other')"}
LABELLED = ("Cryptic", "Possibly cryptic")


def score(npz, emb):
    return ((emb - npz["mean"]) / npz["scale"]) @ npz["coef"] + float(npz["intercept"][0])


def groups_table(df, acc_new, acc_cur, acc_cur_good):
    """Per group: accepted share for the new check and the two readings of the current check."""
    sets = [("Head: all brain", df["head"]), ("Head: trans-thalamic (HC plane)", df["thalamic"])]
    for sp in ("Trans-cerebellum", "Trans-ventricular", "Other"):
        sets.append((f"Head: brain, {sp}", df["head"] & (df["subplane"] == sp)))
    sets += [("NON-HEAD: all (incl. Other)", ~df["head"]), ("NON-HEAD: without Other", ~df["head"] & ~df["is_other"])]
    for c in ("Fetal abdomen", "Fetal femur", "Fetal thorax", "Maternal cervix", "Other"):
        sets.append((f"Non-head: {c}", df["cls"] == c))
    rows = []
    for name, m in sets:
        m = m.to_numpy()
        p = df["patient"].to_numpy()[m]
        r = {"group": name, "n_images": int(m.sum()), "n_patients": int(len(set(p)))}
        for key, acc in (("new", acc_new), ("current_not_refused", acc_cur), ("current_good_only", acc_cur_good)):
            x = rate(acc[m], p)
            r[key] = x["value"]; r[key + "_lo"], r[key + "_hi"] = x["ci95"]; r[key + "_k"] = x["k"]
        rows.append(r)
    return pd.DataFrame(rows)


def auc_ci(s, head, patients, n_boot=1000, seed=0):
    groups = [np.flatnonzero(patients == p) for p in np.unique(patients)]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        h = head[idx]
        if h.any() and (~h).any():
            vals.append(auc(s[idx][h], s[idx][~h]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def tile(path, lines, w=320, h=240):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    k = min(w / img.shape[1], h / img.shape[0])
    img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), np.uint8)
    y0, x0 = (h - img.shape[0]) // 2, (w - img.shape[1]) // 2
    canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    pad = np.full((58, w, 3), 255, np.uint8)
    for j, t in enumerate(lines):
        cv2.putText(pad, t[:46], (4, 15 + 18 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([canvas, pad])


def montage(rows, path, title, cols=4):
    tiles = [tile(r["path"], r["lines"]) for r in rows]
    while tiles and len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.putText(grid, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), grid)


def hc18_block(out, variants):
    """Does the new check still accept HC18 head scans (the data the app was built on)?

    The committed HC18 split cannot be re-created here (the grouped split gives different sizes with these library
    versions), so the scans the current check was fitted on are identified directly: its reference file holds the
    embeddings of the 781 train+val scans, and each of the 999 scans is matched against it."""
    from hcml.splitting import patient_ids

    ref = Reference.load("artifacts/ood_reference.npz")
    idx = pd.read_csv(CACHE / "hc18_train_index.csv")
    emb = np.load(CACHE / "hc18_train_emb.npy")
    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    best = (unit @ ref.feats.T).max(axis=1)
    in_ref = best > 0.999
    assert int(in_ref.sum()) == len(ref.feats) == 781, f"matched {int(in_ref.sum())} of {len(ref.feats)} reference scans"
    pids = np.array(patient_ids(idx["filename"], idx["pixel size(mm)"]))
    ts_idx = pd.read_csv(CACHE / "hc18_testset_index.csv")
    ts_emb = np.load(CACHE / "hc18_testset_emb.npy")
    sets = {"HC18 held-out 218 (never in the current check's reference, never used to fit the new check)":
                (emb, ~in_ref, pids),
            "HC18 unlabeled test_set, 335 (never used by either check)": (ts_emb, np.ones(len(ts_emb), bool), np.arange(len(ts_emb))),
            "HC18 training_set, all 999 (current check is optimistic here: 781 are in its own reference)":
                (emb, np.ones(len(emb), bool), pids)}
    res = {}
    for k, (e, m, g) in sets.items():
        d = np.array([ref.distance(x) for x in e[m]])
        g = g[m]
        row = {"n": int(m.sum()), "current_not_refused": rate(d <= ref.reject, g),
               "current_good_only": rate(d <= ref.warn, g)}
        for name, z in variants.items():
            sc = score(z, e[m])
            for tg in RECALL_TARGETS:
                row[f"new_{name}_at_{int(tg * 100)}"] = rate(sc >= float(z[f"threshold_{int(tg * 100)}"]), g)
            row[f"new_{name}_score_median"] = float(np.median(sc))
        res[k] = row
    res["_note"] = {"scans_matched_to_current_reference": int(in_ref.sum()),
                    "split_reproduction": "re-running grouped_split(seed 42) here gives 639/160/200, not the committed 618/163/218"}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_head_check")
    args = ap.parse_args()
    out = Path(args.out)

    df, emb = load_fetal()
    sp = pd.read_csv(out / "splits.csv")
    assert list(sp["image"]) == list(df["image"])
    df["split"] = sp["split"].values
    te = (df["split"] == "test").to_numpy()
    t = df[te].reset_index(drop=True)
    e = emb[te]
    head = t["head"].to_numpy()
    pat = t["patient"].to_numpy()

    ref = Reference.load("artifacts/ood_reference.npz")
    dist = np.array([ref.distance(x) for x in e])
    acc_cur, acc_cur_good = dist <= ref.reject, dist <= ref.warn
    pipe = pd.read_csv(out / "test_pipeline.csv").set_index("image").loc[t["image"]].reset_index()
    parity = float(np.abs(pipe["distance"].to_numpy() - dist).max())
    assert parity < 5e-3, f"embedding path differs from Predictor by {parity}"

    report = {"test_images": int(te.sum()), "test_patients": int(t["patient"].nunique()),
              "heads": int(head.sum()), "thalamic": int(t["thalamic"].sum()),
              "distance_parity_max_abs_diff_vs_Predictor": parity,
              "majority_baseline": {"always_say_not_head_accuracy": float(1 - head.mean()), "head_share": float(head.mean())}}
    variants = {k: np.load(out / f"head_classifier_{k}.npz") for k in VARIANTS}
    out_rows, per_image = {}, t[["image", "path", "cls", "subplane", "machine", "patient", "head", "is_other"]].copy()
    per_image["current_distance"] = dist
    per_image["current_not_refused"] = acc_cur
    for c in ("ga_weeks", "half_days", "badge", "level", "verdict_No", "verdict_Not sure", "verdict_Yes"):
        per_image[c] = pipe[c].values
    ga_ok = pipe["ga_weeks"].notna().to_numpy()
    label_ok = (pipe["verdict_No"].isin(LABELLED) | pipe["verdict_Not sure"].isin(LABELLED)).to_numpy()

    for name, z in variants.items():
        s = score(z, e)
        per_image[f"score_{name}"] = s
        block = {}
        for tg in RECALL_TARGETS:
            thr = float(z[f"threshold_{int(tg * 100)}"])
            acc_new = s >= thr
            tab = groups_table(t, acc_new, acc_cur, acc_cur_good)
            comb = acc_new & acc_cur                                    # new check AND current check
            nh = ~head
            sel = {"non-head, all": nh, "non-head, without Other": nh & ~t["is_other"].to_numpy()}
            for c in ("Fetal abdomen", "Fetal femur", "Fetal thorax", "Maternal cervix", "Other"):
                sel[c] = (t["cls"] == c).to_numpy()
            combo = {}
            for k, m in sel.items():
                p = pat[m]
                combo[k] = {"n": int(m.sum()),
                            "current_alone_gets_age": rate(ga_ok[m], p), "combined_gets_age": rate((ga_ok & comb)[m], p),
                            "current_alone_gets_cryptic_or_possibly": rate(label_ok[m], p),
                            "combined_gets_cryptic_or_possibly": rate((label_ok & comb)[m], p)}
            hm = head
            combo["heads"] = {"n": int(hm.sum()),
                              "current_alone_gets_age": rate(ga_ok[hm], pat[hm]),
                              "combined_gets_age": rate((ga_ok & comb)[hm], pat[hm]),
                              "current_alone_gets_a_verdict": rate((pipe["verdict_No"] != "Inconclusive").to_numpy()[hm], pat[hm]),
                              "combined_gets_a_verdict": rate(((pipe["verdict_No"] != "Inconclusive").to_numpy() & comb)[hm], pat[hm])}
            decision_acc_new = float(((acc_new == head)).mean())
            decision_acc_cur = float(((acc_cur == head)).mean())
            block[f"head_accept_target_{int(tg * 100)}"] = {
                "threshold": thr, "groups": tab.round(4).to_dict("records"), "combined_with_current": combo,
                "decision_accuracy": {"new": decision_acc_new, "current_not_refused": decision_acc_cur,
                                      "majority_baseline": report["majority_baseline"]["always_say_not_head_accuracy"],
                                      "new_ci95": boot_ci((acc_new == head), pat),
                                      "current_ci95": boot_ci((acc_cur == head), pat)}}
            if tg == PRIMARY_TARGET:
                per_image[f"accept_{name}"] = acc_new
                per_image[f"combined_{name}"] = comb
        block["auc_new"] = {"value": auc(s[head], s[~head]), "ci95": auc_ci(s, head, pat)}
        block["auc_current"] = {"value": auc(-dist[head], -dist[~head]), "ci95": auc_ci(-dist, head, pat)}
        bm = {}
        for mach, g in t.groupby("machine"):
            m = (t["machine"] == mach).to_numpy()
            thr = float(z[f"threshold_{int(PRIMARY_TARGET * 100)}"])
            bm[mach] = {"heads": int((m & head).sum()), "new_head_accepted": float((s[m & head] >= thr).mean()) if (m & head).any() else None,
                        "current_head_accepted": float(acc_cur[m & head].mean()) if (m & head).any() else None,
                        "nonhead": int((m & ~head).sum()), "new_nonhead_accepted": float((s[m & ~head] >= thr).mean()),
                        "current_nonhead_accepted": float(acc_cur[m & ~head].mean())}
        block["by_machine_primary"] = bm
        out_rows[name] = block
    report["variants"] = out_rows
    report["hc18"] = hc18_block(out, variants)
    json.dump(report, open(out / "test_report.json", "w"), indent=2)
    per_image.drop(columns=["path"]).to_csv(out / "test_images.csv", index=False)

    # mistakes in both directions, variant A at the primary target
    z = variants["A_with_other"]
    thr = float(z[f"threshold_{int(PRIMARY_TARGET * 100)}"])
    s = per_image["score_A_with_other"].to_numpy()

    def lines(r):
        return [f"{r['cls']} / {r['subplane'] if r['head'] else '-'} | {r['machine']}",
                f"new score {r['score_A_with_other']:+.2f} (thr {thr:+.2f})",
                f"current d={r['current_distance']:.3f} {'accepted' if r['current_not_refused'] else 'REFUSED'}"]
    fa = per_image[(~per_image["head"]) & (s >= thr)].sort_values("score_A_with_other", ascending=False)
    fa = fa.groupby("cls", group_keys=False).head(3).head(12)
    montage([{**r, "lines": lines(r)} for _, r in fa.iterrows()], out / "examples_new_false_accepts.png",
            "NEW check: non-head images it wrongly accepts (highest scores)")
    fr = per_image[per_image["head"] & (s < thr)]
    fr = pd.concat([fr[fr["subplane"] == "Trans-thalamic"].sort_values("score_A_with_other").head(8),
                    fr[fr["subplane"] != "Trans-thalamic"].sort_values("score_A_with_other").head(4)])
    montage([{**r, "lines": lines(r)} for _, r in fr.iterrows()], out / "examples_new_false_refusals.png",
            "NEW check: head images it wrongly refuses (lowest scores, thalamic first)")
    print("done ->", out)


if __name__ == "__main__":
    main()
