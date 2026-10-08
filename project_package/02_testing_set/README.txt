TESTING SET
===========
The actual test images (copied, unchanged) and, for each dataset, a CSV with the label and the checks' decisions.

images/barcelona_fetal_planes_test/   2,478 FETAL_PLANES_DB images from 336 patients (618 head, of which 332 trans-thalamic; 1,860 not head)
images/hc18_heldout_218/              218 HC18 scans the current model and the head check never used (heads only)
images/hc18_test_set_335/             335 HC18 test_set scans, never used by either (heads only; no published head circumference)
barcelona_fetal_planes_test.csv, hc18_heldout_218.csv, hc18_test_set_335.csv   one row per image
MANIFEST_SHA256.txt                   SHA-256 of every image, to verify a copy
leak_check.json                       machine-readable result of the check below

CSV columns (all three): file_name, dataset, label (head / not head), original_plane, then
  head_check_score    score of the retrained head-view check (higher = more head-like)
  head_check_95       accepted / refused at the 95% operating point (threshold +0.9857); the app default
  head_check_98       accepted / refused at the 98% operating point (threshold -1.0875)
  current_check_distance and current_check   the EXISTING image check: accepted unless distance > 0.2249
  combined_95, combined_98                   what the app does: accepted only if BOTH checks accept
The Barcelona CSV also has brain_subplane, patient_id, us_machine; the held-out CSV has group_id (linked-scan heuristic),
head_circumference_mm, hadlock_age_weeks and age_band; the test_set CSV has an app-estimated age (NOT ground truth).
Labels: head = Fetal brain, all sub-planes (trans-thalamic is the head-circumference measurement plane); not head =
abdomen, femur, thorax, maternal cervix, Other (which may contain a few heads). Every HC18 image is a head.

NO TRAINING OR VALIDATION IMAGE IS IN THIS FOLDER. Check run by scripts/package_check_no_leak.py on 2026-10-08:
  - FETAL_PLANES_DB: files in folder == CSV rows == 2478: True
  - HC18_heldout: files in folder == CSV rows == 218: True
  - HC18_test_set: files in folder == CSV rows == 335: True
  - FETAL_PLANES test file names also in training/validation: 0
  - HC18 held-out file names also in training/validation: 0
  - HC18 test_set file names that equal an HC18 training/validation name (name re-use between releases; pixels decide): 269
  - FETAL_PLANES patients on both sides: 0
  - HC18 linked-scan groups on both sides (held-out vs training/validation): 0
  - training+validation images hashed: 10702
  - FETAL_PLANES_DB: test images with the same pixels as a training/validation image: 0
  - FETAL_PLANES_DB: test images with the same file bytes as a training/validation image: 0
  - HC18_heldout: test images with the same pixels as a training/validation image: 0
  - HC18_heldout: test images with the same file bytes as a training/validation image: 0
  - HC18_test_set: test images with the same pixels as a training/validation image: 0
  - HC18_test_set: test images with the same file bytes as a training/validation image: 0
  - RESULT: PASS: no training or validation image is in the testing set

Notes: the HC18 test_set re-uses file names of the HC18 training set (000_HC.png is in both releases), so for it the name
overlap is expected and the pixel comparison is what proves the images differ. Every test image was compared by decoded
pixels and by file bytes against every training and validation image read from the original datasets.
The HC18 held-out group check compares linked-scan groups from the same heuristic used for training; that heuristic could
not be reproduced exactly (628 groups here vs 585 recorded), see 03_results and the project README.

Verify a copy:  cd 02_testing_set && sha256sum -c MANIFEST_SHA256.txt
This folder is also distributed as 02_testing_set.zip (next to this folder; its SHA-256 is in 02_testing_set.zip.sha256).
Data: FETAL_PLANES_DB (CC BY 4.0) and HC18 (CC BY 4.0). See 05_share_with_group/NOTICE.md.
