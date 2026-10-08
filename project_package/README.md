# project_package: head-view check, organised outputs

**Research prototype. Not validated on cryptic pregnancies. Not for medical use.** Branch `claude/laughing-carson-vaacz0`.

This folder collects everything from the "is this a fetal head?" work (two studies, then wiring into the app). Everything was
**copied**; the original study folders (`eval_*`), the app, the model and the thresholds are unchanged.

| Folder | What it contains | Size on disk | In git? | Who needs it |
|---|---|---|---|---|
| `01_training_set/` | CSV lists of the training and validation images (names, labels, patient/group, split) and a README with how to download the public datasets and rebuild the training set. No images. | 978 KB | yes | Anyone refitting or auditing the classifier |
| `02_testing_set/` | The 3,031 test images (Barcelona 2,478, HC18 held-out 218, HC18 test_set 335), one CSV per dataset with labels and each check's decision at 95% and 98%, SHA-256 manifest and the no-training-image check result. | 473.4 MB | files yes; `images/` no (git-ignored) | Reviewers and group members who want to re-test |
| `02_testing_set.zip` | The folder above as one zip (+ .sha256). | 474.0 MB | no (git-ignored; SHA-256 yes) | Group members, markers |
| `03_results/` | Final evaluation tables (CSV), one-page RESULTS_SUMMARY.md, example images, safety-failure lists, app screenshots (abdomen, femur, thorax, trans-thalamic, 1_2HC.png). | 12.5 MB | yes | Whoever writes the report or paper |
| `04_code_and_models/` | Scripts to reproduce training and evaluation, the saved head classifier and thresholds, requirements.txt, exact commands in README.txt. | 380 KB | yes | Developers |
| `05_share_with_group/` | Summary, citations and licences, NOTICE.md, results summary and a copy of the testing zip. | 474.1 MB | files yes; the zip copy no (git-ignored) | All group members |

**Total size on disk: 1,435.3 MB.** Size that goes into git (everything except the test images and the zips): **14.7 MB**.
The test images and zips are git-ignored (`.gitignore` in this folder) because GitHub rejects files over 100 MB; share the zip
directly. Verify it with `02_testing_set.zip.sha256` (SHA-256 `17f2f7bcd9d75ccc...`) or, per image,
`02_testing_set/MANIFEST_SHA256.txt`. Largest single file: `02_testing_set.zip` (474.0 MB); it exists
twice (here and in `05_share_with_group/`). No full dataset copies are included.

## Start here
- Group member: `05_share_with_group/SUMMARY.md`.
- Report writer: `03_results/RESULTS_SUMMARY.md`.
- Re-test: unzip `02_testing_set.zip`, read `02_testing_set/README.txt`.
- Reproduce: `04_code_and_models/README.txt`.

## How it was checked
- No training or validation image is in the testing set: PASS: no training or validation image is in the testing set (details in `02_testing_set/README.txt`).
- Decisions in the CSVs use the shipped classifier and thresholds (`04_code_and_models/models/`), with scores that match the study to within 0.0001.

## Data and licences
FETAL_PLANES_DB (Burgos-Artizzu et al., Sci Rep 10:10200, 2020, CC BY 4.0, doi:10.5281/zenodo.3904280) and HC18 (van den Heuvel
et al., PLoS ONE 13(8):e0200412, 2018, CC BY 4.0). Attribution text: `05_share_with_group/NOTICE.md`.
