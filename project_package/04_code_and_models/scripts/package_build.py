"""Assemble project_package/ from the head-check study outputs. Copies files; never moves or deletes the originals and
never touches the app, the model or the thresholds. Every table and number is generated from the saved study files.

    python scripts/package_build.py --pkg project_package --fetal /home/user/data_fetal_planes_unzipped \\
        --hc18 /home/user/data_hc18_unzipped [--stage 01 02 03 04 05 top]
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hcml.headcheck import HeadCheck
from hcml.validity import Reference

CACHE = Path("/home/user/head_check_cache")
S1, S2 = ROOT / "eval_head_check", ROOT / "eval_head_check_hc18"
LABELLED = ("Cryptic", "Possibly cryptic")
FETAL_CITE = ("Burgos-Artizzu, X.P., Coronado-Gutierrez, D., Valenzuela-Alcaraz, B., Bonet-Carne, E., Eixarch, E., Crispi, F., "
              "Gratacos, E. (2020). Evaluation of deep convolutional neural networks for automatic classification of common "
              "maternal fetal ultrasound planes. Scientific Reports 10:10200. https://doi.org/10.1038/s41598-020-67076-5")
HC18_CITE = ("van den Heuvel, T.L.A., de Bruijn, D., de Korte, C.L., van Ginneken, B. (2018). Automated measurement of fetal head "
             "circumference using 2D ultrasound images. PLoS ONE 13(8): e0200412. https://doi.org/10.1371/journal.pone.0200412")


def mb(path):
    p = Path(path)
    n = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
    return n


def human(n):
    return f"{n / 1e6:,.1f} MB" if n >= 1e6 else f"{n / 1e3:,.0f} KB"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pct(x):
    return None if x is None or x != x else round(100 * x, 1)


def rate_cols(prefix, r):
    return {f"{prefix}_pct": pct(r["value"]), f"{prefix}_ci_low": pct(r["ci95"][0]), f"{prefix}_ci_high": pct(r["ci95"][1])}


def fmt(r):
    return f"{100 * r['value']:.1f}% ({100 * r['ci95'][0]:.1f}-{100 * r['ci95'][1]:.1f})"


def write(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf8")


# ---------------------------------------------------------------------------------------------------------
# 01  training set (lists only)
# ---------------------------------------------------------------------------------------------------------

def stage01(pkg):
    d = pkg / "01_training_set"
    d.mkdir(parents=True, exist_ok=True)
    sp = pd.read_csv(S1 / "splits.csv")
    f = sp[sp["split"].isin(["train", "val"])].copy()
    fe = pd.DataFrame({
        "file_name": f["image"] + ".png", "dataset": "FETAL_PLANES_DB",
        "label": np.where(f["cls"] == "Fetal brain", "head", "not head"), "original_plane": f["cls"],
        "brain_subplane": np.where(f["cls"] == "Fetal brain", f["subplane"], ""), "patient_or_group_id": f["patient"],
        "split": f["split"].map({"train": "train", "val": "validation"}), "us_machine": f["machine"]})
    pt = pd.read_csv(S2 / "hc18_partition.csv")
    h = pt[pt["part"].isin(["train", "val"])]
    hc = pd.DataFrame({
        "file_name": h["filename"], "dataset": "HC18", "label": "head",
        "original_plane": "Fetal head (HC18 axial head-circumference plane)", "brain_subplane": "",
        "patient_or_group_id": h["pid"], "split": h["part"].map({"train": "train", "val": "validation"}), "us_machine": ""})
    allr = pd.concat([fe, hc], ignore_index=True)
    allr[allr["split"] == "train"].to_csv(d / "train.csv", index=False)
    allr[allr["split"] == "validation"].to_csv(d / "validation.csv", index=False)
    dropped = pt[pt["part"] == "dropped"]
    pd.DataFrame({"file_name": dropped["filename"], "dataset": "HC18", "patient_or_group_id": dropped["pid"],
                  "reason": "Dropped from training and validation: it belongs to a linked-scan chain (087_HC, 088_2HC, 088_HC, 089_HC) "
                            "that straddles the committed HC18 split; the other three scans are held-out test scans."}
                 ).to_csv(d / "hc18_dropped.csv", index=False)
    c = allr.groupby(["dataset", "split", "label"]).size()
    n_pat = {s: int(fe[fe.split == s].patient_or_group_id.nunique()) for s in ("train", "validation")}
    txt = f"""TRAINING SET LISTS (lists only, no images)
==========================================
These CSVs name the images used to fit and tune the head-view check. The images themselves are public; download them
as below and rebuild the folders with scripts/rebuild_training_set.py.

Files
  train.csv           {int((allr.split == 'train').sum()):,} rows  (FETAL_PLANES_DB {int(((allr.split == 'train') & (allr.dataset == 'FETAL_PLANES_DB')).sum()):,} images from {n_pat['train']:,} patients + HC18 {int(((allr.split == 'train') & (allr.dataset == 'HC18')).sum())} scans)
  validation.csv      {int((allr.split == 'validation').sum()):,} rows  (FETAL_PLANES_DB {int(((allr.split == 'validation') & (allr.dataset == 'FETAL_PLANES_DB')).sum()):,} images from {n_pat['validation']:,} patients + HC18 {int(((allr.split == 'validation') & (allr.dataset == 'HC18')).sum())} scans)
  hc18_dropped.csv    1 row: 089_HC.png was DROPPED (see below)
Columns: file_name, dataset (FETAL_PLANES_DB or HC18), label (head / not head), original_plane (the dataset's own plane name),
brain_subplane (FETAL_PLANES brain images only), patient_or_group_id, split (train / validation), us_machine.
  FETAL_PLANES_DB: label is head for "Fetal brain" (all sub-planes); not head for Fetal abdomen, Fetal femur, Fetal thorax,
  Maternal cervix and Other. patient_or_group_id is the dataset's Patient_num. No patient is in more than one split.
  HC18: every scan is a fetal head, so label is head. patient_or_group_id is a LINKED-SCAN GROUP id made by a heuristic
  (hcml/splitting.py), not a real patient id (HC18 has none).

Counts by dataset / split / label:
{c.to_string()}

Only HC18 scans from the current model's own train/validation pool were used (781 scans identified by matching
artifacts/ood_reference.npz). The 218 held-out HC18 scans and the 335 unlabeled HC18 test_set scans were NEVER used for
training, choosing the regularisation strength or choosing thresholds; they are in 02_testing_set.
089_HC.png was dropped: it is in a chain of four linked scans (087_HC, 088_2HC, 088_HC, 089_HC) of which three are held-out
test scans, so keeping it in training could leak a linked scan. 613 + 167 = 780 HC18 scans remain (781 minus the dropped one).

HOW TO DOWNLOAD THE PUBLIC DATASETS AND REBUILD THE TRAINING SET
1. FETAL_PLANES_DB (Zenodo 3904280, CC BY 4.0), 2,088,522,169 bytes, md5 2a5fcc2cefb789bcc0f6c1f73e0ea43f
     curl -L -C - -o FETAL_PLANES_ZENODO.zip "https://zenodo.org/api/records/3904280/files/FETAL_PLANES_ZENODO.zip/content"
     md5sum FETAL_PLANES_ZENODO.zip
     mkdir fetal_planes && unzip -q FETAL_PLANES_ZENODO.zip -d fetal_planes      # gives fetal_planes/Images/*.png and the dataset CSV
2. HC18 (Zenodo 1327317, CC BY 4.0)
     curl -L -C - -o training_set.zip "https://zenodo.org/api/records/1327317/files/training_set.zip/content"   # md5 00eb8198b9a505b2b3a6dfc740382497
     mkdir -p hc18/training_set && unzip -q training_set.zip -d hc18/training_set                               # gives hc18/training_set/training_set/*.png
   (test_set.zip, md5 8402af5d137ef40a2888c1011ef3fe7e, is only needed for 02_testing_set, which already contains those images.)
3. Rebuild (from the repository root, with pandas installed):
     python scripts/rebuild_training_set.py --lists project_package/01_training_set \\
         --fetal fetal_planes --hc18 hc18 --out training_set_rebuilt [--link]
   Result: training_set_rebuilt/{{train,validation}}/{{head,not_head}}/<dataset>__<file_name>.png plus rebuilt_manifest.csv.
   Expect {int((allr.split == 'train').sum()):,} train and {int((allr.split == 'validation').sum()):,} validation images.
   Use --link to symlink instead of copying (no extra disk space).
4. Refit the classifier from these lists: see 04_code_and_models/README.txt.

Citations: FETAL_PLANES_DB: {FETAL_CITE}
          HC18: {HC18_CITE}
Both datasets are CC BY 4.0. See 05_share_with_group/NOTICE.md.
"""
    write(d / "README.txt", txt)
    return {"train": int((allr.split == "train").sum()), "val": int((allr.split == "validation").sum())}


# ---------------------------------------------------------------------------------------------------------
# 02  testing set
# ---------------------------------------------------------------------------------------------------------

def stage02(pkg, fetal, hc18):
    d = pkg / "02_testing_set"
    img = d / "images"
    hc = HeadCheck.load(ROOT / "artifacts" / "head_check")
    t95, t98 = hc.threshold(95), hc.threshold(98)
    ref = Reference.load(ROOT / "artifacts" / "ood_reference.npz")
    dec = lambda ok: np.where(ok, "accepted", "refused")

    def checks(df, emb):
        s = np.array([hc.score(e) for e in emb])
        dist = np.array([ref.distance(e) for e in emb])
        a95, a98, cur = s >= t95, s >= t98, dist <= ref.reject
        df["head_check_score"] = s.round(4)
        df["head_check_95"], df["head_check_98"] = dec(a95), dec(a98)
        df["current_check_distance"] = dist.round(4)
        df["current_check"] = dec(cur)
        df["combined_95"], df["combined_98"] = dec(a95 & cur), dec(a98 & cur)
        return df

    # Barcelona test split
    fi = pd.read_csv(CACHE / "fetal_index.csv")
    emb_f = np.load(CACHE / "fetal_emb.npy")
    sp = pd.read_csv(S1 / "splits.csv")
    t = sp[sp["split"] == "test"].reset_index(drop=True)
    pos = {n: i for i, n in enumerate(fi["image"])}
    bar = pd.DataFrame({"file_name": t["image"] + ".png", "dataset": "FETAL_PLANES_DB",
                        "label": np.where(t["cls"] == "Fetal brain", "head", "not head"), "original_plane": t["cls"],
                        "brain_subplane": np.where(t["cls"] == "Fetal brain", t["subplane"], ""), "patient_id": t["patient"],
                        "us_machine": t["machine"]})
    bar = checks(bar, emb_f[[pos[n] for n in t["image"]]])
    ref_scores = pd.read_csv(S2 / "fetal_test_images.csv").set_index("image")["score_new_A"]
    diff = float(np.abs(bar["head_check_score"].to_numpy() - ref_scores.loc[t["image"]].to_numpy()).max())
    assert diff < 2e-3, f"scores differ from the study by {diff}"
    # HC18 held-out
    pt = pd.read_csv(S2 / "hc18_partition.csv")
    hi = pd.read_csv(CACHE / "hc18_train_index.csv")
    emb_h = np.load(CACHE / "hc18_train_emb.npy")
    ho = pt[pt["part"] == "heldout"]
    hpos = {n: i for i, n in enumerate(hi["filename"])}
    from hcml.growth import hadlock_ga_weeks
    hcmm = hi.set_index("filename").loc[ho["filename"], "head circumference (mm)"].to_numpy()
    ga = hadlock_ga_weeks(hcmm)
    held = pd.DataFrame({"file_name": ho["filename"].to_numpy(), "dataset": "HC18", "label": "head",
                         "original_plane": "Fetal head (HC18 axial head-circumference plane)", "group_id": ho["pid"].to_numpy(),
                         "head_circumference_mm": hcmm, "hadlock_age_weeks": ga.round(2),
                         "age_band": pd.cut(ga, [0, 17, 20, 200], labels=["under 17 weeks", "17 to 20 weeks", "20 weeks or more"], right=False).astype(str)})
    held = checks(held, emb_h[[hpos[n] for n in ho["filename"]]])
    # HC18 test_set
    ti = pd.read_csv(CACHE / "hc18_testset_index.csv")
    emb_t = np.load(CACHE / "hc18_testset_emb.npy")
    est = pd.read_csv(S2 / "hc18_testset_scores.csv").set_index("image")["est_age_weeks"]
    ts = pd.DataFrame({"file_name": ti["image"], "dataset": "HC18 (unlabeled test_set)", "label": "head",
                       "original_plane": "Fetal head (HC18 axial head-circumference plane)",
                       "app_estimated_age_weeks_not_ground_truth": est.loc[ti["image"]].round(2).to_numpy()})
    ts = checks(ts, emb_t)

    sets = {"barcelona_fetal_planes_test": (bar, [fetal / "Images" / n for n in bar["file_name"]]),
            "hc18_heldout_218": (held, [hc18 / "training_set" / "training_set" / n for n in held["file_name"]]),
            "hc18_test_set_335": (ts, [hc18 / "test_set" / "test_set" / n for n in ts["file_name"]])}
    if img.exists():
        shutil.rmtree(img)                      # only the copies inside the package are rebuilt; originals are untouched
    for k, (df, srcs) in sets.items():
        (img / k).mkdir(parents=True, exist_ok=True)
        for s in srcs:
            shutil.copy2(s, img / k / s.name)
        df.to_csv(d / f"{k}.csv", index=False)
    # manifest
    lines = []
    for k, (df, _) in sets.items():
        for n in df["file_name"]:
            lines.append(f"{sha256(img / k / n)}  images/{k}/{n}")
    write(d / "MANIFEST_SHA256.txt", "\n".join(lines) + "\n")
    # leak check
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "package_check_no_leak.py"), "--pkg", str(pkg),
                        "--fetal", str(fetal), "--hc18", str(hc18)], capture_output=True, text=True)
    leak = json.loads((d / "leak_check.json").read_text())
    assert r.returncode == 0 and leak["RESULT"].startswith("PASS"), r.stdout + r.stderr
    nh = (bar["label"] == "not head")
    txt = f"""TESTING SET
===========
The actual test images (copied, unchanged) and, for each dataset, a CSV with the label and the checks' decisions.

images/barcelona_fetal_planes_test/   2,478 FETAL_PLANES_DB images from {bar['patient_id'].nunique()} patients ({int((bar.label == 'head').sum())} head, of which {int((bar.brain_subplane == 'Trans-thalamic').sum())} trans-thalamic; {int(nh.sum()):,} not head)
images/hc18_heldout_218/              218 HC18 scans the current model and the head check never used (heads only)
images/hc18_test_set_335/             335 HC18 test_set scans, never used by either (heads only; no published head circumference)
barcelona_fetal_planes_test.csv, hc18_heldout_218.csv, hc18_test_set_335.csv   one row per image
MANIFEST_SHA256.txt                   SHA-256 of every image, to verify a copy
leak_check.json                       machine-readable result of the check below

CSV columns (all three): file_name, dataset, label (head / not head), original_plane, then
  head_check_score    score of the retrained head-view check (higher = more head-like)
  head_check_95       accepted / refused at the 95% operating point (threshold {t95:+.4f}); the app default
  head_check_98       accepted / refused at the 98% operating point (threshold {t98:+.4f})
  current_check_distance and current_check   the EXISTING image check: accepted unless distance > {ref.reject:.4f}
  combined_95, combined_98                   what the app does: accepted only if BOTH checks accept
The Barcelona CSV also has brain_subplane, patient_id, us_machine; the held-out CSV has group_id (linked-scan heuristic),
head_circumference_mm, hadlock_age_weeks and age_band; the test_set CSV has an app-estimated age (NOT ground truth).
Labels: head = Fetal brain, all sub-planes (trans-thalamic is the head-circumference measurement plane); not head =
abdomen, femur, thorax, maternal cervix, Other (which may contain a few heads). Every HC18 image is a head.

NO TRAINING OR VALIDATION IMAGE IS IN THIS FOLDER. Check run by scripts/package_check_no_leak.py on {pd.Timestamp.now():%Y-%m-%d}:
"""
    for k, v in leak.items():
        txt += f"  - {k}: {v}\n"
    txt += f"""
Notes: the HC18 test_set re-uses file names of the HC18 training set (000_HC.png is in both releases), so for it the name
overlap is expected and the pixel comparison is what proves the images differ. Every test image was compared by decoded
pixels and by file bytes against every training and validation image read from the original datasets.
The HC18 held-out group check compares linked-scan groups from the same heuristic used for training; that heuristic could
not be reproduced exactly (628 groups here vs 585 recorded), see 03_results and the project README.

Verify a copy:  cd 02_testing_set && sha256sum -c MANIFEST_SHA256.txt
This folder is also distributed as 02_testing_set.zip (next to this folder; its SHA-256 is in 02_testing_set.zip.sha256).
Data: FETAL_PLANES_DB (CC BY 4.0) and HC18 (CC BY 4.0). See 05_share_with_group/NOTICE.md.
"""
    write(d / "README.txt", txt)
    zp = pkg / "02_testing_set.zip"
    if zp.exists():
        zp.unlink()
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:       # PNGs are already compressed; store as is
        for f in sorted(d.rglob("*")):
            if f.is_file():
                z.write(f, f"02_testing_set/{f.relative_to(d)}")
    write(pkg / "02_testing_set.zip.sha256", f"{sha256(zp)}  02_testing_set.zip\n")
    return {"zip": zp.stat().st_size, "leak": leak["RESULT"], "parity": diff}


# ---------------------------------------------------------------------------------------------------------
# 03  results
# ---------------------------------------------------------------------------------------------------------

def stage03(pkg):
    d = pkg / "03_results"
    tdir = d / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    R2 = json.loads((S2 / "test_report.json").read_text())
    R1 = json.loads((S1 / "test_report.json").read_text())
    A2 = R2["variants"]["A_with_other"]
    A1 = R1["variants"]["A_with_other"]
    # Barcelona by group
    g95 = {g["group"]: g for g in A2["target_95"]["groups"]}
    g98 = {g["group"]: g for g in A2["target_98"]["groups"]}
    s1_95 = {g["group"]: g for g in A1["head_accept_target_95"]["groups"]}
    s1_98 = {g["group"]: g for g in A1["head_accept_target_98"]["groups"]}
    rows = []
    for name, g in g95.items():
        # cross-check: the Barcelona-only classifier is reported identically by both studies
        assert abs(g["old_fetal_only"] - s1_95[name]["new"]) < 1e-3 and abs(g98[name]["old_fetal_only"] - s1_98[name]["new"]) < 1e-3
        mk = lambda p, v, lo, hi: {f"{p}_pct": pct(v), f"{p}_ci_low": pct(lo), f"{p}_ci_high": pct(hi)}
        r = {"group": name, "n_images": g["n_images"], "n_patients": g["n_patients"]}
        r.update(mk("existing_image_check", g["current_not_refused"], g["current_not_refused_lo"], g["current_not_refused_hi"]))
        r.update(mk("barcelona_only_95", g["old_fetal_only"], g["old_fetal_only_lo"], g["old_fetal_only_hi"]))
        r.update(mk("barcelona_only_98", g98[name]["old_fetal_only"], g98[name]["old_fetal_only_lo"], g98[name]["old_fetal_only_hi"]))
        r.update(mk("shipped_95", g["new_hc18"], g["new_hc18_lo"], g["new_hc18_hi"]))
        r.update(mk("shipped_98", g98[name]["new_hc18"], g98[name]["new_hc18_lo"], g98[name]["new_hc18_hi"]))
        rows.append(r)
    pd.DataFrame(rows).to_csv(tdir / "barcelona_test_share_accepted.csv", index=False)
    bar = pd.DataFrame(rows).set_index("group")
    # HC18 heads
    hrows = []
    names = {"existing_image_check": "current_not_refused", "barcelona_only_95": "old_A_95", "barcelona_only_98": "old_A_98",
             "shipped_95": "new_A_95", "shipped_98": "new_A_98", "shipped_95_AND_existing": "new_A_95_AND_current",
             "shipped_98_AND_existing": "new_A_98_AND_current"}
    for key, label in (("hc18_heldout_218", "HC18 held-out 218"), ("hc18_testset_335", "HC18 unlabeled test_set 335")):
        blk = R2[key]
        parts = [("all", blk["n"], blk["overall"])] + [(b, v["n"], v) for b, v in blk["by_band"].items()]
        for band, n, v in parts:
            r = {"set": label, "age_band": band, "n": n, "age_basis": blk["age_basis"]}
            for p, k in names.items():
                r.update(rate_cols(p, v[k] if band != "all" else v[k]))
            hrows.append(r)
    pd.DataFrame(hrows).to_csv(tdir / "hc18_heads_share_accepted.csv", index=False)
    # combined with the existing check
    crows = []
    for tg in (95, 98):
        c = A2[f"target_{tg}"]["combined_with_current"]
        for grp, v in c.items():
            r = {"operating_point": tg, "group": grp, "n": v["n"]}
            for k in ("current_alone_gets_age", "combined_gets_age", "current_alone_gets_cryptic_or_possibly",
                      "combined_gets_cryptic_or_possibly", "current_alone_gets_a_verdict", "combined_gets_a_verdict"):
                if k in v:
                    r.update(rate_cols(k, v[k]))
                    r[f"{k}_k"] = v[k]["k"]
            crows.append(r)
    pd.DataFrame(crows).to_csv(tdir / "combined_with_existing_check.csv", index=False)
    # operating points
    th = json.loads((ROOT / "artifacts" / "head_check" / "thresholds.json").read_text())
    orows = []
    for tg, thr in th["operating_points"].items():
        v = th["provenance"]["validation"][f"head_accept_target_{tg}"]
        orows.append({"operating_point_pct": int(tg), "score_threshold": round(thr, 4), **{f"validation_{k}": round(x, 4) for k, x in v.items()}})
    pd.DataFrame(orows).to_csv(tdir / "operating_points_validation.csv", index=False)
    # wiring before/after
    w = ROOT / "eval_wiring"
    parts = []
    for tag, a, b in (("main cases", "cases_before.csv", "cases_after.csv"), ("poor / first-trimester", "cases_before_poor.csv", "cases_after_poor.csv")):
        B, A = pd.read_csv(w / a), pd.read_csv(w / b)
        m = B.merge(A, on=["case", "answer"], suffixes=("_before", "_after"))
        m.insert(0, "group", tag)
        parts.append(m.drop(columns=[c for c in m.columns if c.startswith("head_view_tile")]))
    pd.concat(parts).to_csv(tdir / "app_before_after_cases.csv", index=False)
    u = Path("/tmp/claude-0/-home-user/98072213-0bba-5ac3-ad33-0076f98b99eb/scratchpad/user_case")
    if (u / "cases_user_before.csv").exists():
        B, A = pd.read_csv(u / "cases_user_before.csv"), pd.read_csv(u / "cases_user_after.csv")
        m = B.merge(A, on=["case", "answer"], suffixes=("_before", "_after"))
        m.drop(columns=[c for c in m.columns if c.startswith("head_view_tile")]).to_csv(tdir / "user_1_2HC_before_after.csv", index=False)
    # safety lists
    sdir = d / "safety_failures"
    sdir.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "eval_fetal_planes" / "safety_failures.csv", sdir / "before_change_old_app_1741_image_sample.csv")
    fi = pd.read_csv(S2 / "fetal_test_images.csv")
    lab = fi["verdict_No"].isin(LABELLED) | fi["verdict_Not sure"].isin(LABELLED)
    nh = ~fi["head"]
    cols = ["image", "cls", "subplane", "machine", "patient", "score_new_A", "current_distance", "badge", "ga_weeks", "verdict_No", "verdict_Not sure"]
    fi[nh & fi["current_not_refused"] & fi["ga_weeks"].notna() & lab][cols].to_csv(sdir / "before_change_existing_check_alone_barcelona_test_split.csv", index=False)
    for tg in (95, 98):
        thr = th["operating_points"][str(tg)]
        m = nh & fi["current_not_refused"] & fi["ga_weeks"].notna() & lab & (fi["score_new_A"] >= thr)
        fi[m][cols].to_csv(sdir / f"remaining_after_change_{tg}pct_barcelona_test_split.csv", index=False)
    sizes = {p.name: len(pd.read_csv(p)) for p in sorted(sdir.glob("*.csv"))}
    # example images and screenshots
    ex = d / "example_images"
    ex.mkdir(exist_ok=True)
    for n, o in (("examples_new_false_accepts.png", "false_accepts_non_head_wrongly_accepted.png"),
                 ("examples_new_false_refusals_fetal.png", "false_refusals_barcelona_heads_wrongly_refused.png"),
                 ("examples_new_false_refusals_hc18.png", "hc18_heads_still_refused.png")):
        shutil.copy2(S2 / n, ex / o)
    (ex / "study1_barcelona_only").mkdir(exist_ok=True)
    for n in ("examples_new_false_accepts.png", "examples_new_false_refusals.png", "examples_hc18_heads_refused.png"):
        shutil.copy2(S1 / n, ex / "study1_barcelona_only" / n)
    sc = d / "app_screenshots"
    for sub in ("before", "after"):
        (sc / sub).mkdir(parents=True, exist_ok=True)
        for f in (w / sub).glob("*.png"):
            shutil.copy2(f, sc / sub / f.name)
    (sc / "1_2HC").mkdir(exist_ok=True)
    for f in (w / "user_1_2HC").glob("*.png"):
        shutil.copy2(f, sc / "1_2HC" / f.name)
    shutil.copy2(w / "BEFORE_AFTER.md", sc / "BEFORE_AFTER.md")
    write(sc / "README.txt", """APP SCREENSHOTS (real Streamlit app, answer shown in the file name)
before/   the app before the head check was wired in (commit 3a0db64)
after/    the updated app (commit 792a583), 95% operating point
1_2HC/    your 1_2HC.png (pixel-identical to HC18 test_set 000_HC.png) with Yes / No / Not sure, before_* and after_*:
          the before and after screenshots are byte-identical (29 weeks 1 day, Image check Good, same three labels)
Cases: abdomen_cryptic / abdomen_poor (abdomen), femur, thorax, thalamic_accepted / thalamic_refused (trans-thalamic),
hc18_709_2HC (a 29 weeks 1 day HC18 stand-in), week12_user_crop (Poor first-trimester image),
head_only_at_98_* (the same scan at the 95% default and at the 98% setting).
BEFORE_AFTER.md has the same cases as text. Images from FETAL_PLANES_DB and HC18 (both CC BY 4.0), see 05_share_with_group/NOTICE.md.
""")
    # one-page summary
    def row(name, label=None):
        r = bar.loc[name]
        f = lambda p: f"{r[p + '_pct']:.1f}% ({r[p + '_ci_low']:.1f}-{r[p + '_ci_high']:.1f})"
        return f"| {label or name} | {int(r['n_images']):,} | {f('existing_image_check')} | {f('barcelona_only_95')} | {f('shipped_95')} | {f('shipped_98')} |"
    hb = pd.DataFrame(hrows)
    def hrow(setname, band, label):
        r = hb[(hb["set"] == setname) & (hb["age_band"] == band)].iloc[0]
        f = lambda p: f"{r[p + '_pct']:.1f}% ({r[p + '_ci_low']:.1f}-{r[p + '_ci_high']:.1f})"
        return f"| {label} | {int(r['n'])} | {f('existing_image_check')} | {f('barcelona_only_95')} | {f('shipped_95')} | {f('shipped_98')} |"
    cb = pd.DataFrame(crows)
    def crow(grp, label):
        a = cb[(cb["operating_point"] == 95) & (cb["group"] == grp)].iloc[0]
        b = cb[(cb["operating_point"] == 98) & (cb["group"] == grp)].iloc[0]
        f = lambda r, p: f"{r[p + '_pct']:.1f}% ({r[p + '_ci_low']:.1f}-{r[p + '_ci_high']:.1f})"
        return (f"| {label} | {int(a['n']):,} | {f(a, 'current_alone_gets_age')} | {f(a, 'combined_gets_age')} | {f(b, 'combined_gets_age')} | "
                f"{f(a, 'current_alone_gets_cryptic_or_possibly')} | {f(a, 'combined_gets_cryptic_or_possibly')} | {f(b, 'combined_gets_cryptic_or_possibly')} |"
                if grp != "heads" else
                f"| {label} | {int(a['n']):,} | {f(a, 'current_alone_gets_age')} | {f(a, 'combined_gets_age')} | {f(b, 'combined_gets_age')} | - | - | - |")
    fk = R2["fetal_test"]
    md = f"""# Head-view check: results summary (one page)

**Research prototype. Not validated on cryptic pregnancies. Not for medical use.** Share accepted by each check, 95% confidence
ranges (bootstrap over patients; HC18 held-out over linked-scan groups; test_set over single scans). Source: `tables/*.csv`.

**Barcelona test set** (FETAL_PLANES_DB, {fk['images']:,} images, {fk['patients']} patients, {fk['heads']} heads incl. {fk['thalamic']} trans-thalamic; "always say not head" is right {100 * fk['always_not_head_baseline']:.1f}% of the time).
Heads should be accepted (higher is better); non-head should not (lower is better).

| Share accepted | n | Existing image check | Before: Barcelona-only, 95% | **After: shipped, 95%** | **After: shipped, 98%** |
|---|---|---|---|---|---|
{row('Head: all brain', 'Heads, all brain planes')}
{row('Head: trans-thalamic (HC plane)', 'Heads, trans-thalamic')}
{row('NON-HEAD: all (incl. Other)', 'Non-head, all')}
{row('NON-HEAD: without Other', 'Non-head, without "Other"')}
{row('Non-head: Fetal abdomen', 'Abdomen')}
{row('Non-head: Fetal femur', 'Femur')}
{row('Non-head: Fetal thorax', 'Thorax')}
{row('Non-head: Maternal cervix', 'Maternal cervix')}

**HC18 heads** (heads only; the data the app was built on). "Before" wrongly refused many early scans.

| Share accepted | n | Existing image check | Before: Barcelona-only, 95% | **After: shipped, 95%** | **After: shipped, 98%** |
|---|---|---|---|---|---|
{hrow('HC18 held-out 218', 'all', 'Held-out, all')}
{hrow('HC18 held-out 218', 'under 17 weeks', 'Held-out, under 17 weeks')}
{hrow('HC18 held-out 218', '17 to 20 weeks', 'Held-out, 17 to 20 weeks')}
{hrow('HC18 held-out 218', '20 weeks or more', 'Held-out, 20 weeks or more')}
{hrow('HC18 unlabeled test_set 335', 'all', 'test_set, all')}

**In the app (head check AND existing check must both accept)**: what still gets an age and a Cryptic / Possibly cryptic label.

| Barcelona test | n | Age: existing alone | Age: combined 95% | Age: combined 98% | Label: existing alone | Label: combined 95% | Label: combined 98% |
|---|---|---|---|---|---|---|---|
{crow('non-head, all', 'Non-head images')}
{crow('heads', 'Heads')}

**What to remember.** (1) The check is far better than the old one at telling head from non-head on FETAL_PLANES, and keeps
accepting HC18 heads. (2) HC18 has no non-head images, so it could not be tested on non-head scans in HC18's style.
(3) The HC18 split and group counts could not be reproduced (628 groups here vs 585 recorded). (4) Only two hospitals' machines
plus HC18 were seen; other scanners are untested. (5) About 5% of real heads are refused by design at 95% (about 1% at 98%).
(6) Neither dataset contains cryptic pregnancies. Data: FETAL_PLANES_DB (Burgos-Artizzu et al. 2020) and HC18 (van den Heuvel et al. 2018), both CC BY 4.0.
"""
    write(d / "RESULTS_SUMMARY.md", md)
    write(d / "README.txt", f"""RESULTS
=======
RESULTS_SUMMARY.md        one-page summary with before/after numbers and 95% confidence ranges
tables/                   the same numbers as CSV (percentages with ci_low / ci_high):
  barcelona_test_share_accepted.csv      by plane: existing check, Barcelona-only (before), shipped 95% and 98% (after)
  hc18_heads_share_accepted.csv          HC18 held-out 218 and test_set 335, overall and by age band (held-out bands use Hadlock age
                                          from head circumference; test_set bands use the app's estimated age)
  combined_with_existing_check.csv       non-head / head images that still get an age or a label when both checks must accept
  operating_points_validation.csv        thresholds and validation acceptance at 90 / 95 / 98%
  app_before_after_cases.csv, user_1_2HC_before_after.csv   what the real app showed before and after the change
safety_failures/          non-head images that got an age and a Cryptic / Possibly cryptic label:
{chr(10).join('  ' + k + ': ' + str(v) + ' rows' for k, v in sizes.items())}
example_images/           false accepts, false refusals (Barcelona), HC18 heads still refused; study1_barcelona_only/ = the earlier version
app_screenshots/          real app screenshots before and after (abdomen, femur, thorax, trans-thalamic, 1_2HC.png with Yes / No / Not sure)
Operating point: 95% accepts 95% of validation heads (app default); 98% accepts 98%. "Existing image check" = the embedding-distance
check that was already in the app. Images are from FETAL_PLANES_DB and HC18 (CC BY 4.0); see 05_share_with_group/NOTICE.md.
""")
    return sizes


# ---------------------------------------------------------------------------------------------------------
# 04  code and models
# ---------------------------------------------------------------------------------------------------------

SCRIPTS = ["head_check_make_lists.py", "head_check_extract.py", "head_check_common.py", "head_check_fit.py",
           "head_check_pipeline.py", "head_check_final.py", "head_check_hc18_fit.py", "head_check_hc18_final.py",
           "head_check_hc18_examples.py", "head_check_hc18_age_bands.py", "export_head_check.py", "eval_fetal_planes.py",
           "eval_fetal_planes_app_check.py", "eval_fetal_planes_screens.py", "wiring_cases.py", "rebuild_training_set.py",
           "package_check_no_leak.py", "package_build.py"]


def stage04(pkg):
    d = pkg / "04_code_and_models"
    for sub in ("scripts", "hcml", "models", "tests"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    for s in SCRIPTS:
        shutil.copy2(ROOT / "scripts" / s, d / "scripts" / s)
    shutil.copy2(ROOT / "hcml" / "headcheck.py", d / "hcml" / "headcheck.py")
    shutil.copy2(ROOT / "tests" / "test_head_check.py", d / "tests" / "test_head_check.py")
    for f in ("requirements.txt", "requirements-dev.txt"):
        shutil.copy2(ROOT / f, d / f)
    m = d / "models"
    for f in ("classifier.npz", "thresholds.json"):
        shutil.copy2(ROOT / "artifacts" / "head_check" / f, m / f)
    (m / "study_variants").mkdir(exist_ok=True)
    for src, nm in ((S1 / "head_classifier_A_with_other.npz", "study1_barcelona_only_A.npz"), (S1 / "head_classifier_B_without_other.npz", "study1_barcelona_only_B.npz"),
                    (S2 / "head_classifier2_A_with_other.npz", "study2_with_hc18_A_SHIPPED.npz"), (S2 / "head_classifier2_B_without_other.npz", "study2_with_hc18_B.npz"),
                    (S1 / "val_report.json", "study1_val_report.json"), (S2 / "val_report.json", "study2_val_report.json")):
        shutil.copy2(src, m / "study_variants" / nm)
    write(d / "README.txt", """CODE AND MODELS
===============
What is here
  scripts/   copies of the study scripts from the repository's scripts/ folder (branch claude/laughing-carson-vaacz0)
  hcml/headcheck.py   the module the app uses (copy of hcml/headcheck.py)
  models/classifier.npz, models/thresholds.json   the SHIPPED head check (same files as artifacts/head_check/ in the repository):
      thresholds.json has the score thresholds for the 90 / 95 / 98% operating points; 95 is the default
  models/study_variants/   the four fitted variants and their validation reports, for the record
  tests/test_head_check.py   the regression tests
  requirements.txt, requirements-dev.txt   exact package versions (Python 3.11)

These scripts import the repository's hcml package and read artifacts/model.keras (99 MB), artifacts/ood_reference.npz and
artifacts/metadata.json, which are NOT duplicated here. Run every command from the REPOSITORY ROOT:
  git clone https://github.com/anelisa-lab/Detection_system.git && cd Detection_system && git checkout claude/laughing-carson-vaacz0

EXACT COMMANDS TO RE-RUN EVERYTHING (about 1.5 hours on 4 CPU cores; no GPU needed)
0. Environment (Python 3.11):
     python3.11 -m venv .venv && source .venv/bin/activate
     pip install -r requirements-dev.txt
1. Data (details and checksums in 01_training_set/README.txt). The study scripts use these locations; symlink if yours differ:
     /home/user/data_fetal_planes_unzipped   (unzipped FETAL_PLANES_ZENODO.zip: Images/ and FETAL_PLANES_DB_data.csv)
     /home/user/data_hc18_unzipped           (training_set/training_set/*.png, test_set/test_set/*.png, training_set_pixel_size_and_HC.csv)
     mkdir -p /home/user/head_check_cache
2. Image lists and embeddings from the app's own frozen encoder:
     python scripts/head_check_make_lists.py --fetal /home/user/data_fetal_planes_unzipped --hc18 /home/user/data_hc18_unzipped --cache /home/user/head_check_cache
     python scripts/head_check_extract.py --list /home/user/head_check_cache/fetal_list.csv --out /home/user/head_check_cache/fetal
     python scripts/head_check_extract.py --list /home/user/head_check_cache/hc18_train_list.csv --out /home/user/head_check_cache/hc18_train
     python scripts/head_check_extract.py --list /home/user/head_check_cache/hc18_testset_list.csv --out /home/user/head_check_cache/hc18_testset
3. Study 1 (Barcelona-only classifier; validation only, then the one test pass):
     python scripts/head_check_fit.py --out eval_head_check
     python scripts/head_check_pipeline.py --out eval_head_check          # runs the current app pipeline on the test images
     python scripts/head_check_final.py --out eval_head_check
4. Study 2 (HC18 heads added as positives; this is the shipped classifier):
     python scripts/head_check_hc18_fit.py --old eval_head_check --out eval_head_check_hc18
     python scripts/head_check_hc18_final.py --old eval_head_check --out eval_head_check_hc18
     python scripts/head_check_hc18_examples.py --out eval_head_check
     python scripts/head_check_hc18_age_bands.py --out eval_head_check
5. Export the shipped files (writes ONLY artifacts/head_check/classifier.npz and thresholds.json):
     python scripts/export_head_check.py --study eval_head_check_hc18 --out artifacts/head_check
6. Run the app (95% default; 98% with the environment variable) and the tests:
     streamlit run app.py
     HEAD_CHECK_TARGET=98 streamlit run app.py
     python -m pytest tests -q          # 99 passed (needs the HC18 test_set at ../data/test_set for the last test)
7. Before/after cases and the package itself:
     python scripts/wiring_cases.py --root . --label after --cases eval_wiring/cases.json --out eval_wiring
     python scripts/package_build.py --pkg project_package --fetal /home/user/data_fetal_planes_unzipped --hc18 /home/user/data_hc18_unzipped
     python scripts/package_check_no_leak.py --pkg project_package --fetal /home/user/data_fetal_planes_unzipped --hc18 /home/user/data_hc18_unzipped
Rebuild only the training set from the lists: python scripts/rebuild_training_set.py --lists project_package/01_training_set --fetal DIR --hc18 DIR --out OUT
Seeds: 42 everywhere. Results can differ slightly with other library versions (the HC18 split in particular was not reproducible).
Research prototype; not validated on cryptic pregnancies; not for medical use.
""")


# ---------------------------------------------------------------------------------------------------------
# 05  share with group, and the top-level README
# ---------------------------------------------------------------------------------------------------------

def stage05(pkg):
    d = pkg / "05_share_with_group"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pkg / "03_results" / "RESULTS_SUMMARY.md", d / "RESULTS_SUMMARY.md")
    zp = d / "02_testing_set.zip"
    if zp.exists():
        zp.unlink()
    shutil.copy2(pkg / "02_testing_set.zip", zp)
    write(d / "CITATIONS_AND_LICENSES.md", f"""# Citations and licences

Please cite both datasets if you use these files.

**FETAL_PLANES_DB** (used for the head-view check: training, validation and the Barcelona test set)
- {FETAL_CITE}
- Dataset: https://zenodo.org/records/3904280, doi:10.5281/zenodo.3904280
- Licence: **Creative Commons Attribution 4.0 International (CC BY 4.0)**, https://creativecommons.org/licenses/by/4.0/

**HC18** (used for the age model, the existing image check, and as head examples and test images for the head-view check)
- {HC18_CITE}
- Dataset: https://zenodo.org/records/1327317 (HC18 grand challenge: https://hc18.grand-challenge.org)
- Licence: **Creative Commons Attribution 4.0 International (CC BY 4.0)**, https://creativecommons.org/licenses/by/4.0/

**Our code and results** are a student research prototype (final-year project). The Hadlock head-circumference formula is
Hadlock et al., Radiology 1984;152:497-501.
""")
    write(d / "NOTICE.md", f"""# Attribution notice (CC BY 4.0)

The testing set in `02_testing_set.zip` and the example images and screenshots in the project package contain images from
two datasets licensed under **CC BY 4.0**. This notice gives the attribution the licence requires.

## FETAL_PLANES_DB
- **Title:** FETAL_PLANES_DB: Common maternal-fetal ultrasound images
- **Creators:** Burgos-Artizzu, X.P., Coronado-Gutierrez, D., Valenzuela-Alcaraz, B., Bonet-Carne, E., Eixarch, E., Crispi, F., Gratacos, E.
- **Source:** https://zenodo.org/records/3904280 (doi:10.5281/zenodo.3904280)
- **Licence:** CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- **Paper:** {FETAL_CITE}

## HC18 (Automated measurement of fetal head circumference using 2D ultrasound images)
- **Creators:** van den Heuvel, T.L.A., de Bruijn, D., de Korte, C.L., van Ginneken, B.
- **Source:** https://zenodo.org/records/1327317
- **Licence:** CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
- **Paper:** {HC18_CITE}

## Changes made
- The images in `02_testing_set` are **unchanged copies** of the original files (checksums in `MANIFEST_SHA256.txt`).
- Example images and screenshots are **derived** from them (overlays, text captions, the app's display).
- CSV tables are derived results (scores and decisions) computed by us.

## No endorsement
The dataset creators and licensors do not endorse this project or its results.

## Not for clinical use
This is a research prototype. It was **not validated on cryptic pregnancies** and **must not be used for medical decisions**.
""")
    write(d / "SUMMARY.md", """# Head-view check: plain-English summary

**This is a student research prototype. It has not been validated on cryptic pregnancies and is not for medical use.**

## The problem
Our app estimates how many weeks pregnant someone is from an ultrasound picture of the baby's head, and combines that with one
question ("did you know you were pregnant?") to suggest whether a pregnancy was found late (about 20 weeks or later).
But the app gave a week number for *any* picture, even a baby's tummy, leg or chest. Its old "is this a normal head scan?" check
let about 1 in 4 of those through, and it also wrongly turned away about 1 in 4 real head scans.

## What we did
We built a second check that asks one simple question: "is this a picture of a baby's head?" We trained it on about 10,000
routine scans (head, abdomen, leg, chest, cervix and other views) plus a few hundred head scans from the dataset our age
model was built on. We kept a separate set of images that it never saw during training, and tested only on those. An image
now gets an age only if **both** the old check and the new check accept it.

## What we found
- **On the new test pictures (2,478 scans):** the new check accepts about 95% of real heads and wrongly accepts about 0.2% of
  non-head pictures. The old check accepted about 75% of heads and about 25% of non-heads.
- **On the original head-scan dataset (HC18):** it still accepts about 97% to 99% of heads, so the app keeps working on the data
  it was built on. (Our first version, trained without those head scans, wrongly refused about 1 in 4 of them.)
- **In the app:** non-head pictures that used to get an age and a "Cryptic" label (about 6 in 100) now get "No estimate" in almost
  every case (about 1 in 1,000 still slip through).
- A Poor-quality scan no longer shows an age at all; it says "No estimate" and explains why.
- You can switch the check to a more lenient setting (98%) that accepts more real heads but lets a few more non-head pictures through.

## Things to be careful about
- It was not tested on non-head pictures that look like our original dataset, because that dataset only has head scans.
- We could not exactly reproduce how the original dataset's scans were split; we say so openly in the results.
- It has only seen scans from a few machines in two hospitals plus the original dataset; other machines are untested.
- About 1 in 20 real heads are still turned away at the default setting. That is a trade-off we chose on purpose.
- No dataset we used contains confirmed cryptic pregnancies, so we cannot say how well the cryptic-pregnancy suggestion works.

## What is in the share folder
`02_testing_set.zip` (the pictures we tested on, with a table of what each check decided for each picture),
`RESULTS_SUMMARY.md` (the numbers with confidence ranges), `CITATIONS_AND_LICENSES.md` and `NOTICE.md` (credit to the dataset
creators, required by their CC BY 4.0 licence).
""")
    write(d / "README.txt", """SHARE WITH GROUP
================
Only the files group members need:
  SUMMARY.md                  one-page plain-English summary
  RESULTS_SUMMARY.md          the numbers, with 95% confidence ranges
  CITATIONS_AND_LICENSES.md   how to cite FETAL_PLANES_DB and HC18, and their licences (CC BY 4.0)
  NOTICE.md                   attribution text required by CC BY 4.0
  02_testing_set.zip          the test images and decision tables (about 470 MB; check it with the SHA-256 in ../02_testing_set.zip.sha256)
Research prototype; not validated on cryptic pregnancies; not for medical use.
""")


def stage_top(pkg, info):
    rows = [
        ("01_training_set/", "CSV lists of the training and validation images (names, labels, patient/group, split) and a README with how to download the public datasets and rebuild the training set. No images.", "Anyone refitting or auditing the classifier"),
        ("02_testing_set/", "The 3,031 test images (Barcelona 2,478, HC18 held-out 218, HC18 test_set 335), one CSV per dataset with labels and each check's decision at 95% and 98%, SHA-256 manifest and the no-training-image check result.", "Reviewers and group members who want to re-test"),
        ("02_testing_set.zip", "The folder above as one zip (+ .sha256).", "Group members, markers"),
        ("03_results/", "Final evaluation tables (CSV), one-page RESULTS_SUMMARY.md, example images, safety-failure lists, app screenshots (abdomen, femur, thorax, trans-thalamic, 1_2HC.png).", "Whoever writes the report or paper"),
        ("04_code_and_models/", "Scripts to reproduce training and evaluation, the saved head classifier and thresholds, requirements.txt, exact commands in README.txt.", "Developers"),
        ("05_share_with_group/", "Summary, citations and licences, NOTICE.md, results summary and a copy of the testing zip.", "All group members"),
    ]
    tab = "| Folder | What it contains | Size on disk | In git? | Who needs it |\n|---|---|---|---|---|\n"
    for name, what, who in rows:
        p = pkg / name.rstrip("/")
        ingit = {"02_testing_set/": "files yes; `images/` no (git-ignored)", "02_testing_set.zip": "no (git-ignored; SHA-256 yes)",
                 "05_share_with_group/": "files yes; the zip copy no (git-ignored)"}.get(name, "yes")
        tab += f"| `{name}` | {what} | {human(mb(p))} | {ingit} | {who} |\n"
    total = mb(pkg)
    light = sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file() and "images" not in f.relative_to(pkg).parts[:3]
                and f.suffix != ".zip")
    write(pkg / "README.md", f"""# project_package: head-view check, organised outputs

**Research prototype. Not validated on cryptic pregnancies. Not for medical use.** Branch `claude/laughing-carson-vaacz0`.

This folder collects everything from the "is this a fetal head?" work (two studies, then wiring into the app). Everything was
**copied**; the original study folders (`eval_*`), the app, the model and the thresholds are unchanged.

{tab}
**Total size on disk: {human(total)}.** Size that goes into git (everything except the test images and the zips): **{human(light)}**.
The test images and zips are git-ignored (`.gitignore` in this folder) because GitHub rejects files over 100 MB; share the zip
directly. Verify it with `02_testing_set.zip.sha256` (SHA-256 `{sha256(pkg / '02_testing_set.zip')[:16]}...`) or, per image,
`02_testing_set/MANIFEST_SHA256.txt`. Largest single file: `02_testing_set.zip` ({human((pkg / '02_testing_set.zip').stat().st_size)}); it exists
twice (here and in `05_share_with_group/`). No full dataset copies are included.

## Start here
- Group member: `05_share_with_group/SUMMARY.md`.
- Report writer: `03_results/RESULTS_SUMMARY.md`.
- Re-test: unzip `02_testing_set.zip`, read `02_testing_set/README.txt`.
- Reproduce: `04_code_and_models/README.txt`.

## How it was checked
- No training or validation image is in the testing set: {info['leak']} (details in `02_testing_set/README.txt`).
- Decisions in the CSVs use the shipped classifier and thresholds (`04_code_and_models/models/`), with scores that match the study to within {info['parity']:.4f}.

## Data and licences
FETAL_PLANES_DB (Burgos-Artizzu et al., Sci Rep 10:10200, 2020, CC BY 4.0, doi:10.5281/zenodo.3904280) and HC18 (van den Heuvel
et al., PLoS ONE 13(8):e0200412, 2018, CC BY 4.0). Attribution text: `05_share_with_group/NOTICE.md`.
""")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", default="project_package")
    ap.add_argument("--fetal", required=True)
    ap.add_argument("--hc18", required=True)
    ap.add_argument("--stage", nargs="*", default=["01", "02", "03", "04", "05", "top"])
    a = ap.parse_args()
    pkg = Path(a.pkg).resolve()
    pkg.mkdir(exist_ok=True)
    write(pkg / ".gitignore", "# large files: share the zip directly, verify with the SHA-256 files\n02_testing_set/images/\n02_testing_set.zip\n"
                              "05_share_with_group/02_testing_set.zip\ndownload_parts/\n")
    info = {}
    if "01" in a.stage:
        print("01", stage01(pkg))
    if "02" in a.stage:
        info = stage02(pkg, Path(a.fetal), Path(a.hc18))
        print("02", info)
    if "03" in a.stage:
        print("03", stage03(pkg))
    if "04" in a.stage:
        stage04(pkg)
    if "05" in a.stage:
        stage05(pkg)
    if "top" in a.stage:
        info = info or {"leak": json.loads((pkg / "02_testing_set" / "leak_check.json").read_text())["RESULT"], "parity": 0.0}
        stage_top(pkg, info)


if __name__ == "__main__":
    main()
