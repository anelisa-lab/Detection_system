"""Tests for the patient-grouped split and the 20-week screening metrics (no model needed)."""
import numpy as np
import pandas as pd

from hcml.data import add_labels
from hcml.growth import hadlock_hc_mm
from hcml.splitting import grouped_split, patient_ids, screening_20w


def _fake_hc18(n_patients=300, seed=0):
    """A table shaped like the real HC18 CSV: running image numbers (1_HC.png, 8_2HC.png, ...), repeat
    scans carry a 2HC/3HC suffix and sit next to their partner scans. Returns (df, true patient per row)."""
    rng = np.random.default_rng(seed)
    rows, truth, idx = [], [], 0
    for pid in range(n_patients):
        hc = float(rng.uniform(100, 340))
        px = float(rng.uniform(0.05, 0.35))
        for k in range(1, 2 + int(rng.choice([0, 0, 0, 1, 2]))):
            idx += 1
            rows.append((f"{idx}_{'' if k == 1 else k}HC.png", hc + float(rng.normal(0, 3)),
                         px * (1 + float(rng.normal(0, 0.001)))))
            truth.append(pid)
    return pd.DataFrame(rows, columns=["filename", "hc_mm", "pixel_mm"]), np.array(truth)


def test_patient_ids_link_repeat_scans_and_keep_single_scans_apart():
    names = ["1_HC.png", "2_HC.png", "3_2HC.png", "4_HC.png", "5_HC.png", "7_HC.png", "weird.png"]
    px = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    g = patient_ids(names, px)
    assert g[1] == g[2] == g[3]                      # 3_2HC is a repeat scan: linked to both neighbours
    assert len({g[0], g[1], g[4], g[5], g[6]}) == 5  # untouched singles stay separate
    assert patient_ids(["001_HC.png", "001_2HC.png"])[0] == patient_ids(["001_HC.png", "001_2HC.png"])[1]
    assert patient_ids(["4_HC.png", "5_HC.png"], [0.1000, 0.1002])[0] == patient_ids(["4_HC.png", "5_HC.png"], [0.1000, 0.1002])[1]


def test_grouped_split_never_separates_scans_of_one_patient():
    raw, truth = _fake_hc18()
    df = add_labels(raw, (220.0, 290.0))
    out, info = grouped_split(df, 0.2, 0.2, 42)
    out["truth"] = truth
    sides = out.groupby("truth")["split"].nunique()
    assert (sides == 1).all()                         # every real patient sits in exactly one split
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not set(out.loc[out["split"] == a, "patient_id"]) & set(out.loc[out["split"] == b, "patient_id"])
    assert info["patients_in_more_than_one_split"] == 0
    assert set(out["split"]) == {"train", "val", "test"}
    assert abs((out["split"] == "test").mean() - 0.2) < 0.06
    assert abs((out["split"] == "val").mean() - 0.16) < 0.06
    assert set(out.loc[out["split"] == "test", "label"]) == {0, 1, 2}      # every class is in the test split
    assert out["filename"].tolist() == df["filename"].tolist()             # row order kept (features line up)


def test_split_is_reproducible():
    df = add_labels(_fake_hc18()[0], (220.0, 290.0))
    a, _ = grouped_split(df, 0.2, 0.2, 42)
    b, _ = grouped_split(df, 0.2, 0.2, 42)
    assert a["split"].tolist() == b["split"].tolist()


def test_names_without_the_hc18_pattern_each_count_as_one_patient():
    df = add_labels(pd.DataFrame({"filename": [f"img{i}.png" for i in range(100)],
                                  "hc_mm": np.linspace(120, 330, 100)}), (220.0, 290.0))
    out, info = grouped_split(df, 0.2, 0.2, 42)
    assert info["n_patients"] == 100 and info["images_per_split"]["test"] == 20


def test_screening_metrics_perfect_predictions():
    hc = np.array([hadlock_hc_mm(w) for w in (15, 17, 19, 21, 25, 30, 35, 38)])
    r = screening_20w(hc, hc, n_boot=200)
    assert r["n_positive"] == 5 and r["n_negative"] == 3
    assert r["recall_sensitivity"]["value"] == 1.0 and r["specificity"]["value"] == 1.0
    assert r["precision"]["value"] == 1.0 and r["accuracy"]["value"] == 1.0
    assert r["counts"] == {"tp": 5, "fp": 0, "fn": 0, "tn": 3}


def test_screening_metrics_counts_and_baseline():
    hc_true = np.array([hadlock_hc_mm(w) for w in (15, 18, 22, 25, 30, 35)])     # 4 positives
    hc_pred = np.array([hadlock_hc_mm(w) for w in (15, 21, 22, 18, 30, 35)])     # 1 FP, 1 FN
    r = screening_20w(hc_true, hc_pred, n_boot=200)
    assert r["counts"] == {"tp": 3, "fp": 1, "fn": 1, "tn": 1}
    assert abs(r["recall_sensitivity"]["value"] - 0.75) < 1e-9
    assert abs(r["specificity"]["value"] - 0.5) < 1e-9
    assert abs(r["majority_baseline_accuracy"] - 4 / 6) < 1e-9


def test_confidence_interval_brackets_the_estimate_and_groups_are_resampled_together():
    rng = np.random.default_rng(1)
    ga = rng.uniform(14, 38, 200)
    hc_true = np.array([hadlock_hc_mm(w) for w in ga])
    hc_pred = np.array([hadlock_hc_mm(w) for w in ga + rng.normal(0, 2, 200)])
    groups = np.repeat(np.arange(100), 2)
    r = screening_20w(hc_true, hc_pred, groups=groups, n_boot=300)
    lo, hi = r["recall_sensitivity"]["ci95"]
    assert lo <= r["recall_sensitivity"]["value"] <= hi and 0 <= lo <= hi <= 1
    assert r["bootstrap"] == {"resamples": 300, "unit": "patient"}
    assert screening_20w(hc_true, hc_pred, n_boot=50)["bootstrap"]["unit"] == "scan"


def test_screening_metrics_with_no_negatives_do_not_crash():
    hc = np.array([hadlock_hc_mm(w) for w in (25, 30, 35)])
    r = screening_20w(hc, hc, n_boot=50)
    assert r["specificity"]["value"] is None and r["recall_sensitivity"]["value"] == 1.0
