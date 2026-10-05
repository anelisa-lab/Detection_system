# Late-discovery screening aid for cryptic pregnancy: HC18 model and web app

A research prototype that estimates gestational age from a fetal-head ultrasound (HC18 dataset, TensorFlow/Keras
ResNet50) and, from that estimate plus one question, flags pregnancies discovered late (about 20 weeks or later),
the usual definition of cryptic. It is a **screening aid for a clinician to confirm**, not a diagnostic tool: it
does not detect cryptic pregnancy, and HC18 contains no confirmed cryptic pregnancies. Built on the GROUP11 research guide
(`GROUP11_PRJT302_updated_v2`), with Grad-CAM for explanation.

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
shown in the app.

## Web app

```bash
streamlit run app.py          # loads artifacts/; set MODEL_DIR=... for another folder
```

Upload PNG, JPEG, BMP or DICOM scans and answer one question (did you know you were pregnant before this scan?). For
each scan the app shows the preprocessed image, a Grad-CAM overlay with the detected skull, the estimated
gestational age (weeks and days, likely range, trimester, due date and a growth chart), an image-check badge, a
cryptic pregnancy screening result (below), and a short plain-language "What this scan suggests" paragraph. When the
image check is Limited or Poor it shows only a short sentence saying the estimate is unreliable. Several uploads give
a per-image result, an overall average, a CSV and a PDF report.

### Cryptic pregnancy screening result

The scan is the main evidence; it cannot show whether the person knew about the pregnancy, so the app asks one
question above the upload: "Did you know you were pregnant before this scan?" (Yes / No / Not sure, default Not
sure), plus an optional "weeks when you found out" if Yes. No other history is collected. `hcml/screening.py`
(`screen()`, unit-tested in `tests/test_screening.py`) turns the estimate, its range, the image check and the answer
into one result, using the usual definition (cryptic = not recognised until about 20 weeks or later):

| Situation | Result |
|---|---|
| Image check Poor or estimate withheld | Cannot assess ("the scan is not a reliable head view ... If you are worried, please see a clinician regardless.") |
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
