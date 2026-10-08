# Data attribution

The images in `examples/` and the file names in the CSVs come from **FETAL_PLANES_DB** (Zenodo record 3904280,
https://zenodo.org/records/3904280, DOI 10.5281/zenodo.3904280), licensed **CC BY 4.0**.

Burgos-Artizzu, X.P., Coronado-Gutiérrez, D., Valenzuela-Alcaraz, B., Bonet-Carne, E., Eixarch, E., Crispi, F.,
Gratacós, E. Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal
ultrasound planes. Nature Scientific Reports 10, 10200 (2020). https://doi.org/10.1038/s41598-020-67076-5

This folder is an external robustness test of the committed model. Nothing was trained, and `artifacts/` was not
changed. The dataset has no gestational age, so the age estimates here can only be checked for plausibility.
Regenerate with `scripts/eval_fetal_planes.py` (see its docstring), then `scripts/eval_fetal_planes_app_check.py`.
