# Data attribution

Results and example images in this folder are derived from two CC BY 4.0 datasets.

- **FETAL_PLANES_DB**, Zenodo 3904280 (doi:10.5281/zenodo.3904280). Burgos-Artizzu et al., Evaluation of deep
  convolutional neural networks for automatic classification of common maternal fetal ultrasound planes, Sci Rep 10,
  10200 (2020), doi:10.1038/s41598-020-67076-5.
- **HC18**, Zenodo 1327317 (https://zenodo.org/records/1327317). van den Heuvel et al., Automated measurement of fetal
  head circumference using 2D ultrasound images, PLoS ONE 13(8): e0200412 (2018).

This is a study of a possible "is this a fetal head?" check. Nothing here is used by the app. `artifacts/` was not
changed. The classifiers (`head_classifier_*.npz`) are logistic regressions on the committed model's frozen embeddings,
fitted on FETAL_PLANES_DB patients only. Reproduce with the `scripts/head_check_*.py` files (see their docstrings).
