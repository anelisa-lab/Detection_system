# Early detection of cryptic pregnancy: HC18 model and web app

Implements the pipeline in the GROUP11 research guide (`GROUP11_PRJT302_updated_v2`):
a ResNet50 transfer-learning CNN with Grad-CAM, trained on the HC18 fetal head
ultrasound dataset, plus a Streamlit app that serves it.

The model estimates gestational age from a fetal-head ultrasound and, from that estimate plus one question,
suggests whether a pregnancy was recognised late (about 20 weeks or later, the usual definition of cryptic).
It is a research prototype for a clinician to confirm, not a diagnostic tool, and HC18 contains no confirmed
cryptic pregnancies.

All numbers below are from the held-out test split (218 scans) of `artifacts/reports/metrics.json`
(also stored in `artifacts/metadata.json`).

## Results

### 1. Age estimate (the main output)

A regression head predicts head circumference (HC), converted to gestational age with the Hadlock 1984
head-circumference formula (Radiology 152:497-501). The "reference" age is that same formula applied to the
measured HC, not clinical dating.

| Metric | Test (n = 218) |
|---|---|
| Mean absolute error | 10.3 days |
| Median absolute error | 6.8 days |
| RMSE | 15.5 days |
| HC error (MAE) | 17.0 mm |
| R² (gestational age) | 0.864 |
| Within 7 days / within 14 days | 52.8% / 78.9% |
| Range shown in the app | ±17.9 days (80th percentile of validation errors) |

### 2. 20-week screening (is the pregnancy 20 weeks or more?)

Reference and prediction are both Hadlock age from HC. 95% CIs come from a 2,000-resample bootstrap over groups of
linked scans.

| Metric | Value | 95% CI |
|---|---|---|
| Recall (sensitivity) | 78.7% | 69.1-87.2 |
| Precision | 78.7% | 68.7-87.0 |
| Specificity | 79.1% | 69.1-86.9 |
| Accuracy | 78.9% | 71.7-85.3 |

Counts: 85 TP, 23 FP, 23 FN, 87 TN. Majority-class baseline accuracy: 50.5%.

### 3. Three-class stage (early / mid / late, secondary output)

Accuracy **89.0%**, against an always-"early" baseline of **81.7%** (178 of 218 scans are early). This accuracy
equals weighted recall by construction; it is not the screening recall above. Recall is strong only for the
majority class: **mid 42.3% (26 scans)** and **late 35.7% (14 scans)**.

### Safeguards

- Skull locator: mean IoU 0.80 (95.9% of test scans at IoU 0.5 or more).
- Grad-CAM: median overlap with the skull 54.0%; 54.1% of scans have at least half the heatmap on the skull; the
  peak is on the skull in 64.2%.
- Input check cut points: warn 0.194, reject 0.225 (see below).

### Data

999 HC18 scans in 585 linked-scan groups, split 618 / 163 / 218 (train / validation / test), with no group in more
than one split (asserted in `hcml/splitting.py` at training time).

## Setup

```bash
pip install -r requirements.txt     # exact versions; Python 3.11 (see .python-version)
```

Versions are pinned to the set the committed `artifacts/model.keras` was saved and tested with (TensorFlow 2.21.0,
Keras 3.15.1). Developer tools (pytest, playwright) are in `requirements-dev.txt`.

Download HC18 from https://zenodo.org/records/1327317 (or hc18.grand-challenge.org) and
unzip `training_set.zip` so the folder holds `000_HC.png`, `000_HC_Annotation.png`, ...
and `training_set_pixel_size_and_HC.csv` (the CSV may also sit next to the folder):

```
data/hc18/training_set/
```

## Train

```bash
python train.py --data data/hc18/training_set --out artifacts
```

Runs on CPU in a few minutes, because the backbone is frozen and features are computed once.
Options: `--thresholds 220 290` sets the early/mid and mid/late cut points in mm
(defaults are the HC values Hadlock's formula maps to about 24 and 32 weeks; the guide
does not give its numbers, so set them to the ones the group used), `--no-denoise`, `--epochs`.

| Guide spec | Where |
|---|---|
| Resize 224×224, 3-channel, normalise, denoise | `hcml/preprocess.py` |
| HC, ellipse area, perimeter, intensity mean/std, entropy | `hcml/data.py`, written to `reports/dataset_features.csv` (descriptive only) |
| Frozen ImageNet ResNet50 → Dense 256 → Dropout 0.5 → regression head and 3-class softmax head | `hcml/model.py` |
| Grouped 80/20 test split, 20% of train for validation, early stopping patience 10 on val loss | `train.py`, `hcml/splitting.py` |
| Age metrics, 20-week screening with CIs, stage accuracy/precision/F1/specificity | `reports/metrics.json` |
| Confusion matrix, regression scatter, Grad-CAM on 2 correct and 2 misclassified test images | `reports/*.png` |
| DICOM input via pydicom | `hcml/preprocess.py`, used by the app |

Outputs in `artifacts/`: `model.keras`, `locator.keras`, `ood_reference.npz`, `metadata.json`, and `reports/`
(metrics, confusion matrix, training curves, regression scatter, Grad-CAM examples, splits, features; `reports/` is
git-ignored). Head circumference is never a model input. `metadata.json` stores the HC mean/std and the error range
shown in the app. The head-view check files in `artifacts/head_check/` are not produced by `train.py`: they are exported from
the study by `scripts/export_head_check.py` (see "Head-view check").

## Web app

```bash
streamlit run app.py          # loads artifacts/; set MODEL_DIR=... for another folder
```

Upload PNG, JPEG, BMP or DICOM scans and answer one question (did you know you were pregnant before this scan?). For
each scan the app shows the preprocessed image, a Grad-CAM overlay with the detected skull, the estimated
gestational age (weeks and days, likely range, trimester, due date and a growth chart), an image-check badge, a
cryptic pregnancy screening result (below), and a short plain-language "What this scan suggests" paragraph. When the
image check is **Limited** the age is shown with a warning that it is rough. When the verdict is **Cannot assess** (the
image check is Poor, or the head-view check or the image check refuses the image) the page shows **No estimate** with a
short reason, no age headline, no age-derived tiles and no cryptic label. Several uploads give
a per-image result, an overall average, a CSV and a PDF report.

### Cryptic pregnancy screening result

The scan is the main evidence; it cannot show whether the person knew about the pregnancy, so the app asks one
question above the upload: "Did you know you were pregnant before this scan?" (Yes / No / Not sure, default Not
sure), plus an optional "weeks when you found out" if Yes. No other history is collected. `hcml/screening.py`
(`screen()`, unit-tested in `tests/test_screening.py`) turns the estimate, its range, the image check and the answer
into one result, using the usual definition (cryptic = not recognised until about 20 weeks or later):

| Situation | Result |
|---|---|
| Image check Poor, estimate withheld, or refused by the head-view check or the image check | Cannot assess ("the scan is not a reliable head view ... If you are worried, please see a clinician regardless."), with **No estimate** shown instead of an age |
| Found out at about 20 weeks or later (answer Yes with a week of 20 or more) | **Cryptic**, whatever the scan estimate; if the scan suggests under 20 weeks a note says the two disagree |
| Estimate under 20 weeks | Not cryptic by the usual definition |
| 20 weeks or more, answer No | Consistent with a cryptic pregnancy |
| 20 weeks or more, answer Yes (found out earlier or week not given) | Not cryptic: the pregnancy was known |
| 20 weeks or more, Not sure | Possibly cryptic: answer the question to confirm |
| Image check Limited | every verdict is softened to "May be ..." with the reason |
| Range straddles 20 weeks | "Borderline around 20 weeks", both readings shown |

"Not cryptic" results add stage, trimester, development, usual care and when to contact a doctor;
"Cryptic"/"Possibly cryptic" results add a supportive paragraph urging prompt confirmation, dating and
antenatal care. The wording is "suggests" and "consistent with", never a diagnosis; the result, answer and reason go
into the PDF. A clinician must confirm everything.

### Input check (is this a head-circumference view?)

The regressor always returns a number, so every upload is first checked against the training
scans: the mean cosine distance from its ResNet50 embedding to the 5 nearest training embeddings.
(This is one of two checks; see "Head-view check" below. An image is accepted only if both accept it.)
Cut points are calibrated at training time on valid scans never used for fitting (the HC18 test
split plus the unlabeled `test_set` folder next to `training_set`): 95th percentile = "unusual"
(range widened up to 2x), 99.5th percentile = refused (about 0.5% of valid scans). Refused images
get no age and no heatmap, and the app shows a short sentence saying the estimate is unreliable. The user can enter a CRL in mm instead (Robinson & Fleming 1975, 10-84 mm), which is a
manual first-trimester path, not read from the image. Ranges are also widened 1.5x outside 14-36 weeks,
where HC18 is sparse. A hand-built ellipse detector was tried and dropped: it rejected valid HC18
scans and scored a CRL view higher than the typical valid scan.

HC18 holds 1 scan below 12 weeks and 55 between 12 and 13 weeks (by Hadlock age from HC), and
no whole-fetus views at all.

Tests (`python -m pytest tests -q`) cover a valid head view, the week-12 CRL image, a non-ultrasound
photo, DICOM input, unreadable files, the false-rejection rate, the image-check badge and the scan-only
"What this scan suggests" paragraph (shown for a Good image, replaced by a short reliability sentence otherwise).
`tests/test_head_check.py` adds regression tests for the head-view check: its files and operating points (95 default, 98,
the `HEAD_CHECK_TARGET` setting), the rule that both checks must accept, non-head planes (abdomen, femur, thorax) refused with
No estimate and no Cryptic label, HC18-style heads and a trans-thalamic head still accepted, the `crl_week12` and
`week12_user_crop` fixtures, and that a Cannot assess verdict never shows an age headline. The HC18 `test_set` test needs the
folder next to the repository (`../data/test_set`) and is skipped without it.

### Image-check badge (Good / Limited / Poor)

Each accepted scan gets a badge. It drops to **Limited** with one reason and **Poor** with two or more:
the estimate is in the lowest 5% of training ages (under about 13 weeks); the Grad-CAM peak is outside
the detected skull, or under 50% of the heatmap lies on it; no skull-like region is found, or a solid
black box (for example a redaction block) covers part of it; or the embedding distance is above the 90th
percentile of valid scans. A visible warning and a one-line reason are shown. The skull is found by a small
head-localiser (`locator.keras`, trained on HC18 annotation ellipses; mean test IoU 0.80, 95.9% of test scans at IoU 0.5 or more) whose blob is fitted
with an ellipse. The preprocessing itself never masks the image; black boxes seen in the preview come from
the uploaded file.

Calibration on valid HC18 scans: Grad-CAM puts at least half its weight on the skull in only 54.1% of test
scans (median overlap 54.0%, peak on the skull in 64.2%), so the rule downgrades many valid scans to Limited or Poor.
That is a weakness of the heatmap, kept visible on purpose.

### Head-view check (is this a fetal head at all?)

The image check above measures how much a scan *resembles HC18*. That is not the same as "is this a head": on a test of
FETAL_PLANES_DB (12,400 routine screening images, six plane types) it accepted 24.6% of non-head images and refused 24.6%
of real heads. A second, separate check was therefore added (`hcml/headcheck.py`). It is a logistic regression on the 2048-d
embedding of the app's own frozen encoder, fitted on FETAL_PLANES_DB (fetal brain planes = head; abdomen, femur, thorax,
maternal cervix and "other" planes = not head) plus 613 HC18 scans. **An image is accepted only if both checks accept it.**
If the head-view check refuses it, the page shows the same Cannot assess / No estimate result as any refused image, with the
reason "This image does not look like a standard fetal head view".

* Files, separate from the model and the image-check thresholds: `artifacts/head_check/classifier.npz` and
  `artifacts/head_check/thresholds.json` (written by `scripts/export_head_check.py`; nothing else in `artifacts/` was changed).
  Without that folder the app behaves as before.
* **Operating point** (share of validation heads the check is set to accept): **95 by default**, or 98 (accepts more real
  heads, lets a few more non-head images through). Choose it in Settings > Head-view check, or start the app with
  `HEAD_CHECK_TARGET=98 streamlit run app.py`. 90 is also available through the environment variable. No code change is needed.
* The threshold was chosen on validation data only. The FETAL_PLANES split is by `Patient_num` (no patient in two splits). The
  218 held-out HC18 scans and the 335 unlabeled HC18 `test_set` scans were never used for training, choosing C or choosing
  thresholds. Study code and all results: `scripts/head_check_*.py`, `eval_head_check/`, `eval_head_check_hc18/`; the before/after
  cases and screenshots of the wiring are in `eval_wiring/`. Intervals are 95%, bootstrapped over patients (FETAL_PLANES),
  linked-scan groups (HC18 held-out) or single scans (HC18 test_set).

**Barcelona test set** (FETAL_PLANES_DB, 2,478 images from 336 patients; 618 heads, of which 332 trans-thalamic, the HC
measurement plane; always saying "not head" would be right 75.1% of the time):

| Share accepted | Existing image check | Study 1: Barcelona-only, 95% / 98% | **Shipped (study 2): with HC18 heads, 95%** | **Shipped: 98%** |
|---|---|---|---|---|
| Heads, all brain planes | 75.4% (70.2-80.0) | 95.8% (93.8-97.5) / 99.2% (98.3-99.8) | **94.5%** (92.5-96.3) | **98.7%** (97.7-99.7) |
| Trans-thalamic heads | 80.7% (75.1-85.5) | 98.2% (96.4-99.7) / 100.0% | **97.6%** (95.7-99.1) | **100.0%** |
| Non-head, all (incl. Other) | 24.6% (21.0-28.4) | 0.3% (0.1-0.7) / 3.1% (2.2-4.1) | **0.2%** (0.1-0.4) | **1.4%** (0.8-2.0) |
| Non-head, without Other | 24.6% (21.0-28.2) | 0.4% (0.1-0.8) / 3.6% (2.4-4.9) | **0.4%** (0.1-0.8) | **1.7%** (0.9-2.6) |
| Abdomen | 42.5% (33.6-51.2) | 0.0% / 6.7% (2.9-11.5) | **0.0%** | **3.0%** (0.7-6.2) |
| Femur | 32.3% (25.3-39.4) | 0.4% (0.0-1.5) / 3.6% (1.4-6.2) | **0.4%** (0.0-1.5) | **3.2%** (0.9-5.7) |
| Thorax | 29.2% (22.1-36.6) | 0.9% (0.0-2.0) / 4.9% (2.6-7.5) | **0.9%** (0.0-2.0) | **1.5%** (0.3-2.9) |
| Maternal cervix | 4.8% (1.5-9.1) | 0.0% / 0.7% (0.0-1.8) | **0.0%** | **0.4%** (0.0-1.1) |
| Other | 24.6% (18.7-31.4) | 0.2% (0.0-0.6) / 2.5% (1.4-3.9) | **0.0%** | **1.0%** (0.4-1.8) |

Correct head / not-head decisions: 98.5% (95%) and 98.6% (98%), against 75.4% for the existing check. Brain images labelled
"Other" (angled, non-axial heads, n=29) are accepted only 48.3% (28.1-69.2) of the time at 95%.

**HC18 heads** (the data the app was built on; heads only). Study 1, trained on Barcelona data alone, wrongly refused many of
them, mostly early scans, which is why HC18 heads were added as positives:

| Share accepted | Existing image check | Study 1 at 95% / 98% | **Shipped, 95%** | **Shipped, 98%** |
|---|---|---|---|---|
| Held-out 218, all | 99.1% (97.7-100.0) | 76.1% (67.8-84.1) / 85.8% (78.7-92.1) | **97.2%** (94.7-99.5) | **99.1%** (97.6-100.0) |
| ... under 17 weeks (n=61) | 96.7% (91.8-100.0) | 26.2% (16.9-37.5) | **90.2%** (82.7-96.8) | **96.7%** (91.2-100.0) |
| ... 17 to 20 weeks (n=49) | 100.0% | 95.9% (88.7-100.0) | **100.0%** | **100.0%** |
| ... 20 weeks or more (n=108) | 100.0% | 95.4% (91.5-99.0) | **100.0%** | **100.0%** |
| Unlabeled test_set 335, all | 99.4% (98.5-100.0) | 85.1% (81.2-89.0) / 93.1% (90.1-95.8) | **99.4%** (98.5-100.0) | **100.0%** |

Held-out age bands use Hadlock age from head circumference. The test_set has no published head circumference, so its bands
(in `eval_head_check_hc18/test_report.json`) use the app model's own estimated age.

**Combined with the existing check** (Barcelona test; "gets an age" is what the page would show):

| | Existing check alone | Combined, 95% | Combined, 98% |
|---|---|---|---|
| Non-head images that still get an age (n=1,860) | 24.6% (21.0-28.4) | **0.1%** (2 images) | **0.8%** (14 images) |
| ... and a Cryptic or Possibly cryptic label | 5.9% (4.4-7.5) | **0.1%** (2 images) | **0.4%** (7 images) |
| Barcelona heads that get an age (n=618) | 75.4% (70.2-80.0) | 72.8% (67.9-77.3) | 75.2% (70.0-79.7) |
| HC18 held-out heads that get an age (n=218) | 99.1% | 97.2% | 98.6% |
| HC18 test_set heads that get an age (n=335) | 99.4% | 98.8% | 99.4% |

## Limitations

- **Reference age.** The 20-week reference is Hadlock age computed from head circumference, not clinical dating
  (LMP or early-scan dating), so the screening figures measure agreement with an HC-derived age.
- **Cryptic pregnancy is not validated.** HC18 has no confirmed cryptic pregnancies, so the cryptic verdict itself
  has never been tested; the app can only combine an age estimate with the user's answer.
- **Patient grouping is a heuristic.** HC18 has no patient IDs; scans are linked by repeat-scan suffix, neighbouring
  image numbers and pixel size. This is conservative but not ground truth.
- **Mid and late stages rest on very few scans.** Recall of 42.3% (mid) and 35.7% (late) comes from only 26 and 14
  test scans; the 89.0% stage accuracy is driven by the early class (81.7% baseline).
- **Squashed 224×224 input.** Images are resized to a square, which loses scale (pixel size), and head circumference
  in mm depends on it, so some error cannot be removed.
- **Grad-CAM often misses the skull.** The heatmap has at least half its weight on the skull in only about 54% of
  test scans, so the model may be using markers or on-screen text. Treat the heatmap as a sanity check, not an
  explanation.
- **Input-check cut points were calibrated on test scans.** The warn (0.194) and reject (0.225) distances use the
  HC18 test split (plus the unlabeled `test_set` when present), so the check's false-reject rate is not independent.
  Reported age and screening metrics are not affected.
- **HC18 has no non-head images.** The head-view check could not be tested on non-head scans that look like HC18, and it may
  partly have learned "HC18 look means head". The two first-trimester fixtures moved toward acceptance: `week12_user_crop.png`
  is refused by the Barcelona-only version but accepted by the shipped one (the image check still rates it Poor, so the verdict
  is Cannot assess), and `crl_week12.png` stays refused (by the head check at 95%; at 98% only the image check refuses it).
  That is why both checks are required.
- **The HC18 split and group counts could not be reproduced.** `metadata.json` records 585 linked-scan groups and a 618/163/218
  split. With the code in this repository and the HC18 CSV from Zenodo, `patient_ids` gives **628** groups (not 585), re-running
  `grouped_split` with seed 42 gives 639/160/200, and a four-scan chain (087_HC, 088_2HC, 088_HC, 089_HC) straddles the
  committed split. The 218 held-out scans were therefore identified by matching them against `ood_reference.npz`, not by
  re-running the split, and the claim "no linked scan in more than one split" cannot be verified from this repository. Describe
  the grouping as a heuristic.
- **The head-view check is trained on two hospitals' machines plus HC18.** Other scanners are untested. FETAL_PLANES labels are
  not perfect (the most confidently accepted "femur" image shows a skull), a trans-thalamic scan that the existing image check
  refuses stays refused (a scan is accepted only if both checks accept it), and about 5% of real heads are refused by design at
  the 95% point (about 1% at 98%).
- **Not validated on cryptic pregnancies.** Neither dataset contains confirmed cryptic pregnancies; the screening result is a
  rule applied to an age estimate and one answer.
- **Single-centre data and few first-trimester scans.** HC18 has 1 scan below 12 weeks and 55 between 12 and 13
  weeks, and no whole-fetus views; estimates below about 14 weeks are less reliable and may be mis-dated by weeks.
  Scans before about 14 weeks are normally measured by crown-rump length (CRL); the app offers manual CRL entry
  (Robinson & Fleming 1975) but cannot measure it from the image.
- Not validated on non-pregnant cases. The two-class "pregnancy finding present / absent" model (guide Section 8.1)
  needs source-matched non-pregnant scans.

## Workstation UI

A dark, calm imaging-workstation layout (light theme in Settings). Left: brand, navigation (Scan, Batch, Report, About),
and collapsible Settings and Model performance. Centre: the viewer (Original / Preprocessed / Grad-CAM with the skull
outline, a caption bar with file name, size and image-check badge, heatmap opacity and a side-by-side compare toggle).
Right: the estimate with range, trimester, due date and image-check chips; the screening result with its badge and
"why" line; four metric tiles (head circumference, heatmap on skull, image-check score, range width) with gauges; and
the growth chart. A 40-week timeline sits under the viewer, with the stage guide below. Status is always icon plus text
(Good, Limited, Poor, Rejected; Cryptic, Possibly cryptic, Not cryptic, Cannot assess), contrast is at least 4.5:1 in both
themes (`tests/test_ui_contrast.py`), images carry alt text, and the disclaimer is pinned in the footer. Code:
`hcml/ui.py` (cards), `hcml/ui.css` (one stylesheet, palettes are CSS variables), `app.py` (pages).

Batch mode analyses with a progress bar, then shows a summary strip with counts per image-check badge and a paged table
(thumbnail, file, estimate, image check, screening result) that can be filtered by badge or result and sorted; "Open"
shows that scan in the viewer. Tested with all 335 scans of the HC18 `test_set`. CSV export and a PDF report (up to 100
scans) are on the Batch and Report pages. The heatmap-opacity slider lives in the viewer toolbar.

Regenerate the screenshots with `python scripts/screenshots.py [--all]` (needs `pip install playwright` and Chrome, with
the app running).

| Empty state | Good scan (late stage, answer "No") |
|---|---|
| ![Empty state](docs/screens/1_empty_dark.png) | ![Good scan](docs/screens/2_good_scan_dark.png) |

| Light theme | Poor scan |
|---|---|
| ![Light theme](docs/screens/3_good_scan_light.png) | ![Poor scan](docs/screens/4_poor_scan_dark.png) |

| Batch of 12 | Batch of 335 | Mobile |
|---|---|---|
| ![Batch of 12](docs/screens/5_batch_dark.png) | ![Batch of 335](docs/screens/7_batch_335_dark.png) | ![Mobile](docs/screens/6_mobile_dark.png) |

## Deploy on Render

The trained model files in `artifacts/` are committed, so the app runs as-is. Create a Web Service from the repo:

- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
- Python 3.11 is pinned by `.python-version` (TensorFlow does not support the newest Python releases).
- Use a plan with at least 2 GB of RAM; TensorFlow plus ResNet50 will not fit in 512 MB.

## Smoke test without HC18

```bash
python scripts/make_synthetic_hc18.py --out data/synthetic/training_set --n 150
python train.py --data data/synthetic/training_set --out artifacts_smoke
MODEL_DIR=artifacts_smoke streamlit run app.py
```

The synthetic images are drawn ellipses, only for checking the code runs. Do not report results from them.

## Data, licences and citations

* **FETAL_PLANES_DB** (head-view check): Burgos-Artizzu, X.P., Coronado-Gutierrez, D., Valenzuela-Alcaraz, B., Bonet-Carne, E.,
  Eixarch, E., Crispi, F., Gratacos, E. Evaluation of deep convolutional neural networks for automatic classification of common
  maternal fetal ultrasound planes. *Scientific Reports* 10:10200 (2020). https://doi.org/10.1038/s41598-020-67076-5.
  Dataset: https://zenodo.org/records/3904280, doi:10.5281/zenodo.3904280. Licence **CC BY 4.0**. Six regression fixtures in
  `tests/fixtures/` and the example images in `eval_*` folders are taken from it unchanged (see the NOTICE files).
* **HC18** (age model, image check, head-view check positives): van den Heuvel, T.L.A., de Bruijn, D., de Korte, C.L.,
  van Ginneken, B. Automated measurement of fetal head circumference using 2D ultrasound images. *PLoS ONE* 13(8): e0200412
  (2018). Dataset: https://zenodo.org/records/1327317. Licence **CC BY 4.0**.
