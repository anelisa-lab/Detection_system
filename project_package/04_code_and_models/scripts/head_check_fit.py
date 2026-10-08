"""Stage 1: fit the 'is this a fetal head?' classifier on frozen embeddings and choose its threshold on VALIDATION only.

Positive = Fetal brain (all sub-planes). Negative = abdomen, femur, thorax, cervix, Other. Logistic regression on the
2048-d embedding of the committed model's frozen encoder, class-balanced. The test split is not touched here.

    python scripts/head_check_fit.py --out eval_head_check

Two variants are fitted: A trains with "Other" as a negative class, B leaves "Other" out of training (it may contain
heads). The threshold for each is the highest value that still accepts PRIMARY_TARGET (95%) of the validation heads.
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
from sklearn.preprocessing import StandardScaler

from head_check_common import (PRIMARY_TARGET, RECALL_TARGETS, auc, load_fetal, patient_split,
                               threshold_for_recall)
from hcml.validity import Reference

C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def far_at_recall(scores, head, mask_neg, target):
    t = threshold_for_recall(scores[head], target)
    return t, float((scores[mask_neg & ~head] >= t).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="eval_head_check")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, emb = load_fetal()
    df["split"] = patient_split(df, args.seed)
    df[["image", "patient", "cls", "subplane", "machine", "split"]].to_csv(out / "splits.csv", index=False)
    summary = {"split_counts": {s: {"images": int((df.split == s).sum()), "patients": int(df[df.split == s].patient.nunique()),
                                    "heads": int(df[df.split == s]["head"].sum())} for s in ("train", "val", "test")}}
    print(json.dumps(summary["split_counts"], indent=1))

    ref = Reference.load("artifacts/ood_reference.npz")
    dist = np.array([ref.distance(e) for e in emb])
    tr, va = (df.split == "train").to_numpy(), (df.split == "val").to_numpy()
    head = df["head"].to_numpy()
    other = df["is_other"].to_numpy()

    summary["current_check_on_val"] = {"auc": auc(-dist[va & head], -dist[va & ~head])}
    for tgt in RECALL_TARGETS:
        t = float(np.quantile(-dist[va & head], 1 - tgt, method="lower"))
        summary["current_check_on_val"][f"far_at_head_accept_{int(tgt * 100)}"] = float((-dist[va & ~head] >= t).mean())

    results = {}
    for name, use_other in (("A_with_other", True), ("B_without_other", False)):
        fit_rows = tr & (use_other | ~other)
        sc = StandardScaler().fit(emb[fit_rows])
        Xtr, ytr = sc.transform(emb[fit_rows]), head[fit_rows]
        best = None
        table = []
        for C in C_GRID:
            lr = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xtr, ytr)
            s = lr.decision_function(sc.transform(emb[va]))
            # choose C on the negatives the variant knows about; B is not judged on "Other"
            neg = (np.ones(va.sum(), bool) if use_other else ~other[va])
            t, far = far_at_recall(s, head[va], neg, PRIMARY_TARGET)
            a = auc(s[head[va]], s[~head[va] & neg])
            table.append({"C": C, "val_auc": a, f"val_far_at_head_accept_{int(PRIMARY_TARGET * 100)}": far})
            if best is None or far < best[0] - 1e-12:
                best = (far, C, lr)
        far, C, lr = best
        s = lr.decision_function(sc.transform(emb[va]))
        thr = {str(int(tg * 100)): threshold_for_recall(s[head[va]], tg) for tg in RECALL_TARGETS}
        val = {"chosen_C": C, "grid": table, "thresholds_from_val": thr, "val_scores_by_class": {}}
        for tg in RECALL_TARGETS:
            t = thr[str(int(tg * 100))]
            row = {"head_accepted": float((s[head[va]] >= t).mean()),
                   "thalamic_accepted": float((s[df["thalamic"].to_numpy()[va]] >= t).mean()),
                   "nonhead_accepted_all": float((s[~head[va]] >= t).mean()),
                   "nonhead_accepted_without_other": float((s[~head[va] & ~other[va]] >= t).mean())}
            for c in sorted(df.loc[~df["head"], "cls"].unique()):
                m = (df.loc[va, "cls"] == c).to_numpy()
                row[f"accepted_{c}"] = float((s[m] >= t).mean())
            val["val_scores_by_class"][f"head_accept_target_{int(tg * 100)}"] = row
        results[name] = val
        np.savez(out / f"head_classifier_{name}.npz", mean=sc.mean_, scale=sc.scale_, coef=lr.coef_[0],
                 intercept=lr.intercept_, C=C, **{f"threshold_{k}": v for k, v in thr.items()})
        print(f"\n{name}: chosen C={C}  val AUC={table[[r['C'] for r in table].index(C)]['val_auc']:.4f}")
        print(pd.DataFrame(val["val_scores_by_class"]).T.round(3).to_string())
    summary["variants"] = results
    json.dump(summary, open(out / "val_report.json", "w"), indent=2)
    print("\ncurrent check on val:", json.dumps(summary["current_check_on_val"], indent=1))


if __name__ == "__main__":
    main()
