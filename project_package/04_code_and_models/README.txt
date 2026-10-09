CODE AND MODELS
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
     mkdir -p /home/user/head_check_cache      # or set HEAD_CHECK_CACHE to another folder; scripts say so if it is empty
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
     python -m pytest tests -q          # 96 passed, 3 skipped; 99 passed when the HC18 test_set is at ../data/test_set (or set HC18_TEST_SET)
7. Before/after cases and the package itself:
     python scripts/wiring_cases.py --root . --label after --cases eval_wiring/cases.json --out eval_wiring
     python scripts/package_build.py --pkg project_package --fetal /home/user/data_fetal_planes_unzipped --hc18 /home/user/data_hc18_unzipped
     python scripts/package_check_no_leak.py --pkg project_package --fetal /home/user/data_fetal_planes_unzipped --hc18 /home/user/data_hc18_unzipped
Rebuild only the training set from the lists: python scripts/rebuild_training_set.py --lists project_package/01_training_set --fetal DIR --hc18 DIR --out OUT
Seeds: 42 everywhere. Results can differ slightly with other library versions (the HC18 split in particular was not reproducible).
Research prototype; not validated on cryptic pregnancies; not for medical use.
