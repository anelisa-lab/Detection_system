TRAINING SET LISTS (lists only, no images)
==========================================
These CSVs name the images used to fit and tune the head-view check. The images themselves are public; download them
as below and rebuild the folders with scripts/rebuild_training_set.py.

Files
  train.csv           8,055 rows  (FETAL_PLANES_DB 7,442 images from 1,092 patients + HC18 613 scans)
  validation.csv      2,647 rows  (FETAL_PLANES_DB 2,480 images from 364 patients + HC18 167 scans)
  hc18_dropped.csv    1 row: 089_HC.png was DROPPED (see below)
Columns: file_name, dataset (FETAL_PLANES_DB or HC18), label (head / not head), original_plane (the dataset's own plane name),
brain_subplane (FETAL_PLANES brain images only), patient_or_group_id, split (train / validation), us_machine.
  FETAL_PLANES_DB: label is head for "Fetal brain" (all sub-planes); not head for Fetal abdomen, Fetal femur, Fetal thorax,
  Maternal cervix and Other. patient_or_group_id is the dataset's Patient_num. No patient is in more than one split.
  HC18: every scan is a fetal head, so label is head. patient_or_group_id is a LINKED-SCAN GROUP id made by a heuristic
  (hcml/splitting.py), not a real patient id (HC18 has none).

Counts by dataset / split / label:
dataset          split       label   
FETAL_PLANES_DB  train       head        1855
                             not head    5587
                 validation  head         619
                             not head    1861
HC18             train       head         613
                 validation  head         167

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
     python scripts/rebuild_training_set.py --lists project_package/01_training_set \
         --fetal fetal_planes --hc18 hc18 --out training_set_rebuilt [--link]
   Result: training_set_rebuilt/{train,validation}/{head,not_head}/<dataset>__<file_name>.png plus rebuilt_manifest.csv.
   Expect 8,055 train and 2,647 validation images.
   Use --link to symlink instead of copying (no extra disk space).
4. Refit the classifier from these lists: see 04_code_and_models/README.txt.

Citations: FETAL_PLANES_DB: Burgos-Artizzu, X.P., Coronado-Gutierrez, D., Valenzuela-Alcaraz, B., Bonet-Carne, E., Eixarch, E., Crispi, F., Gratacos, E. (2020). Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes. Scientific Reports 10:10200. https://doi.org/10.1038/s41598-020-67076-5
          HC18: van den Heuvel, T.L.A., de Bruijn, D., de Korte, C.L., van Ginneken, B. (2018). Automated measurement of fetal head circumference using 2D ultrasound images. PLoS ONE 13(8): e0200412. https://doi.org/10.1371/journal.pone.0200412
Both datasets are CC BY 4.0. See 05_share_with_group/NOTICE.md.
