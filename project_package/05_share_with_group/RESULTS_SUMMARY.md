# Head-view check: results summary (one page)

**Research prototype. Not validated on cryptic pregnancies. Not for medical use.** Share accepted by each check, 95% confidence
ranges (bootstrap over patients; HC18 held-out over linked-scan groups; test_set over single scans). Source: `tables/*.csv`.

**Barcelona test set** (FETAL_PLANES_DB, 2,478 images, 336 patients, 618 heads incl. 332 trans-thalamic; "always say not head" is right 75.1% of the time).
Heads should be accepted (higher is better); non-head should not (lower is better).

| Share accepted | n | Existing image check | Before: Barcelona-only, 95% | **After: shipped, 95%** | **After: shipped, 98%** |
|---|---|---|---|---|---|
| Heads, all brain planes | 618 | 75.4% (70.2-80.0) | 95.8% (93.8-97.5) | 94.5% (92.5-96.3) | 98.7% (97.7-99.7) |
| Heads, trans-thalamic | 332 | 80.7% (75.1-85.5) | 98.2% (96.4-99.7) | 97.6% (95.7-99.1) | 100.0% (100.0-100.0) |
| Non-head, all | 1,860 | 24.6% (21.0-28.4) | 0.3% (0.1-0.7) | 0.2% (0.1-0.4) | 1.4% (0.8-2.0) |
| Non-head, without "Other" | 989 | 24.6% (21.0-28.2) | 0.4% (0.1-0.8) | 0.4% (0.1-0.8) | 1.7% (0.9-2.6) |
| Abdomen | 134 | 42.5% (33.6-51.2) | 0.0% (0.0-0.0) | 0.0% (0.0-0.0) | 3.0% (0.7-6.2) |
| Femur | 220 | 32.3% (25.3-39.4) | 0.4% (0.0-1.5) | 0.4% (0.0-1.5) | 3.2% (0.9-5.7) |
| Thorax | 346 | 29.2% (22.1-36.6) | 0.9% (0.0-2.0) | 0.9% (0.0-2.0) | 1.5% (0.3-2.9) |
| Maternal cervix | 289 | 4.8% (1.5-9.1) | 0.0% (0.0-0.0) | 0.0% (0.0-0.0) | 0.4% (0.0-1.1) |

**HC18 heads** (heads only; the data the app was built on). "Before" wrongly refused many early scans.

| Share accepted | n | Existing image check | Before: Barcelona-only, 95% | **After: shipped, 95%** | **After: shipped, 98%** |
|---|---|---|---|---|---|
| Held-out, all | 218 | 99.1% (97.7-100.0) | 76.1% (67.8-84.1) | 97.2% (94.7-99.5) | 99.1% (97.6-100.0) |
| Held-out, under 17 weeks | 61 | 96.7% (91.8-100.0) | 26.2% (16.9-37.5) | 90.2% (82.7-96.8) | 96.7% (91.2-100.0) |
| Held-out, 17 to 20 weeks | 49 | 100.0% (100.0-100.0) | 95.9% (88.7-100.0) | 100.0% (100.0-100.0) | 100.0% (100.0-100.0) |
| Held-out, 20 weeks or more | 108 | 100.0% (100.0-100.0) | 95.4% (91.5-99.0) | 100.0% (100.0-100.0) | 100.0% (100.0-100.0) |
| test_set, all | 335 | 99.4% (98.5-100.0) | 85.1% (81.2-89.0) | 99.4% (98.5-100.0) | 100.0% (100.0-100.0) |

**In the app (head check AND existing check must both accept)**: what still gets an age and a Cryptic / Possibly cryptic label.

| Barcelona test | n | Age: existing alone | Age: combined 95% | Age: combined 98% | Label: existing alone | Label: combined 95% | Label: combined 98% |
|---|---|---|---|---|---|---|---|
| Non-head images | 1,860 | 24.6% (21.0-28.4) | 0.1% (0.0-0.3) | 0.8% (0.4-1.2) | 5.9% (4.4-7.5) | 0.1% (0.0-0.3) | 0.4% (0.1-0.7) |
| Heads | 618 | 75.4% (70.2-80.0) | 72.8% (67.9-77.3) | 75.2% (70.0-79.7) | - | - | - |

**What to remember.** (1) The check is far better than the old one at telling head from non-head on FETAL_PLANES, and keeps
accepting HC18 heads. (2) HC18 has no non-head images, so it could not be tested on non-head scans in HC18's style.
(3) The HC18 split and group counts could not be reproduced (628 groups here vs 585 recorded). (4) Only two hospitals' machines
plus HC18 were seen; other scanners are untested. (5) About 5% of real heads are refused by design at 95% (about 1% at 98%).
(6) Neither dataset contains cryptic pregnancies. Data: FETAL_PLANES_DB (Burgos-Artizzu et al. 2020) and HC18 (van den Heuvel et al. 2018), both CC BY 4.0.
