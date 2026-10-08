"""Stage 1 (HC18-augmented): refit the 'is this a fetal head?' classifier with HC18 head scans added as positives.

Fair-test rules:
  * Only HC18 scans the current model already used (its train + validation scans, identified by matching against
    artifacts/ood_reference.npz) may be used. The 218 held-out HC18 scans and the 335 unlabeled test_set are never
    used for training, choosing C, or choosing thresholds.
  * The FETAL_PLANES patient-grouped split is the one from head_check_fit.py (OUT_OLD/splits.csv), so the old and new
    test sets are identical.
  * The threshold is chosen on validation only: the highest value that still accepts 95% of ALL validation heads
    (FETAL_PLANES brain + HC18 validation scans pooled). 90% and 98% operating points are also saved.

    python scripts/head_check_hc18_fit.py --old eval_head_check --out eval_head_check_hc18
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from head_check_common import CACHE, PRIMARY_TARGET, RECALL_TARGETS, auc, load_fetal, threshold_for_recall
from hcml.splitting import patient_ids
from hcml.validity import Reference

C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def hc18_partition(seed: int = 42):
    """Rows of the 999 HC18 scans: 'heldout' (not used by the current model), and the 781 split into train / val."""
    idx = pd.read_csv(CACHE / "hc18_train_index.csv")
    emb = np.load(CACHE / "hc18_train_emb.npy")
    ref = Reference.load("artifacts/ood_reference.npz")
    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    in_ref = (unit @ ref.feats.T).max(axis=1) > 0.999
    assert int(in_ref.sum()) == len(ref.feats) == 781
    idx["pid"] = patient_ids(idx["filename"], idx["pixel size(mm)"])
    idx["ga"] = None
    shared = set(idx.loc[in_ref, "pid"]) & set(idx.loc[~in_ref, "pid"])
    part = pd.Series("heldout", index=idx.index)
    # a linked-scan group that straddles the committed split: keep its train/val scans OUT of training entirely, so
    # no held-out scan has a linked scan in the training pool
    pool = np.flatnonzero(in_ref & ~idx["pid"].isin(shared).to_numpy())
    part.iloc[np.flatnonzero(in_ref & idx["pid"].isin(shared).to_numpy())] = "dropped"
    tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(pool, groups=idx.loc[pool, "pid"]))
    part.iloc[pool[tr]], part.iloc[pool[va]] = "train", "val"
    idx["part"] = part
    return idx, emb, len(shared)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", default="eval_head_check", help="folder with the earlier splits.csv")
    ap.add_argument("--out", default="eval_head_check_hc18")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, emb = load_fetal()
    sp = pd.read_csv(Path(args.old) / "splits.csv")
    assert list(sp["image"]) == list(df["image"])
    df["split"] = sp["split"].values
    hidx, hemb, shared_groups = hc18_partition(args.seed)
    hidx[["filename", "pid", "part"]].to_csv(out / "hc18_partition.csv", index=False)
    counts = hidx["part"].value_counts().to_dict()
    print("HC18 partition:", counts, "| linked-scan groups shared between the 781 and the 218:", shared_groups)

    head, other = df["head"].to_numpy(), df["is_other"].to_numpy()
    tr, va = (df["split"] == "train").to_numpy(), (df["split"] == "val").to_numpy()
    h_tr, h_va = (hidx["part"] == "train").to_numpy(), (hidx["part"] == "val").to_numpy()

    summary = {"hc18_partition": counts, "linked_groups_shared_with_heldout": int(shared_groups),
               "train_counts": {"fetal_images": int(tr.sum()), "fetal_heads": int((tr & head).sum()), "hc18_heads_added": int(h_tr.sum())},
               "val_counts": {"fetal_images": int(va.sum()), "fetal_heads": int((va & head).sum()), "hc18_heads": int(h_va.sum())}}
    print(json.dumps(summary, indent=1))
    results = {}
    for name, use_other in (("A_with_other", True), ("B_without_other", False)):
        fit_f = tr & (use_other | ~other)
        Xtr = np.vstack([emb[fit_f], hemb[h_tr]])
        ytr = np.concatenate([head[fit_f], np.ones(int(h_tr.sum()), bool)])
        sc = StandardScaler().fit(Xtr)
        Xtr = sc.transform(Xtr)
        neg_mask = np.ones(int(va.sum()), bool) if use_other else ~other[va]
        best, table = None, []
        for C in C_GRID:
            lr = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xtr, ytr)
            s_f, s_h = lr.decision_function(sc.transform(emb[va])), lr.decision_function(sc.transform(hemb[h_va]))
            heads_val = np.concatenate([s_f[head[va]], s_h])             # all validation heads, pooled
            t = threshold_for_recall(heads_val, PRIMARY_TARGET)
            far = float((s_f[~head[va] & neg_mask] >= t).mean())
            table.append({"C": C, "val_auc_fetal": auc(s_f[head[va]], s_f[~head[va] & neg_mask]),
                          "val_far_at_95": far, "val_hc18_head_accepted_at_95": float((s_h >= t).mean())})
            if best is None or far < best[0] - 1e-12:
                best = (far, C, lr)
        far, C, lr = best
        s_f, s_h = lr.decision_function(sc.transform(emb[va])), lr.decision_function(sc.transform(hemb[h_va]))
        heads_val = np.concatenate([s_f[head[va]], s_h])
        thr = {str(int(tg * 100)): threshold_for_recall(heads_val, tg) for tg in RECALL_TARGETS}
        val = {"chosen_C": C, "grid": table, "thresholds_from_val": thr, "at_target": {}}
        for tg in RECALL_TARGETS:
            t = thr[str(int(tg * 100))]
            row = {"heads_pooled": float((heads_val >= t).mean()), "fetal_heads": float((s_f[head[va]] >= t).mean()),
                   "fetal_thalamic": float((s_f[df["thalamic"].to_numpy()[va]] >= t).mean()),
                   "hc18_val_heads": float((s_h >= t).mean()),
                   "nonhead_all": float((s_f[~head[va]] >= t).mean()),
                   "nonhead_without_other": float((s_f[~head[va] & ~other[va]] >= t).mean())}
            for c in sorted(df.loc[~df["head"], "cls"].unique()):
                row[f"accepted_{c}"] = float((s_f[(df.loc[va, "cls"] == c).to_numpy()] >= t).mean())
            val["at_target"][f"head_accept_target_{int(tg * 100)}"] = row
        results[name] = val
        np.savez(out / f"head_classifier2_{name}.npz", mean=sc.mean_, scale=sc.scale_, coef=lr.coef_[0],
                 intercept=lr.intercept_, C=C, **{f"threshold_{k}": v for k, v in thr.items()})
        print(f"\n{name}: chosen C={C}")
        print(pd.DataFrame(val["at_target"]).T.round(3).to_string())
    summary["variants"] = results
    json.dump(summary, open(out / "val_report.json", "w"), indent=2)


if __name__ == "__main__":
    main()
