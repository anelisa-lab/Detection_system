# Data attribution

Results and example images in this folder are derived from two CC BY 4.0 datasets.

- **FETAL_PLANES_DB**, Zenodo 3904280 (doi:10.5281/zenodo.3904280). Burgos-Artizzu et al., Sci Rep 10, 10200 (2020),
  doi:10.1038/s41598-020-67076-5.
- **HC18**, Zenodo 1327317. van den Heuvel et al., PLoS ONE 13(8): e0200412 (2018).

Study of a retrained "is this a fetal head?" check (HC18 head scans added as positives). Nothing here is used by the app.
`artifacts/` was not changed. Training used FETAL_PLANES_DB train patients plus 613 HC18 scans from the current model's own
train/validation pool; the 218 HC18 held-out scans and the 335 unlabeled test_set scans were never used for training,
choosing C, or choosing thresholds. Reproduce with `scripts/head_check_hc18_fit.py` then `scripts/head_check_hc18_final.py`.
