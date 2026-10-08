# Before / after: head-view check wired into the app (95% operating point)

Generated from `cases_before*.csv` and `cases_after*.csv` (real Streamlit app via AppTest, see `scripts/wiring_cases.py`).
**Before** = commit 3a0db64 (nothing wired in). **After** = this commit. Screenshots: `before/` and `after/`.

`1_2HC.png` is not in the official HC18 download (its files are zero-padded and repeat scans start at 010_2HC.png). Two official scans that
the app dates at exactly 29 weeks 1 day with image check Good were used as stand-ins; to run your own file:
`python scripts/wiring_cases.py --root . --label mine --cases my_cases.json --out eval_wiring` with a cases file such as
`{"1_2HC": {"path": "C:/path/1_2HC.png", "answers": ["Yes", "No", "Not sure"]}}`.


## main cases

| Case | Answer | Before: age shown / image check / result | After: age shown / image check / result |
|---|---|---|---|
| hc18_standin_709_2HC (709_2HC.png) | Yes | 29 weeks 1 day / Good / Not cryptic: Not cryptic: the pregnancy was known | 29 weeks 1 day / Good / Not cryptic: Not cryptic: the pregnancy was known |
| hc18_standin_709_2HC (709_2HC.png) | No | 29 weeks 1 day / Good / Cryptic: Consistent with a cryptic pregnancy | 29 weeks 1 day / Good / Cryptic: Consistent with a cryptic pregnancy |
| hc18_standin_709_2HC (709_2HC.png) | Not sure | 29 weeks 1 day / Good / Possibly cryptic: Possibly cryptic: answer the question above to confirm | 29 weeks 1 day / Good / Possibly cryptic: Possibly cryptic: answer the question above to confirm |
| hc18_standin_testset_000_HC (000_HC.png) | Yes | 29 weeks 1 day / Good / Not cryptic: Not cryptic: the pregnancy was known | 29 weeks 1 day / Good / Not cryptic: Not cryptic: the pregnancy was known |
| hc18_standin_testset_000_HC (000_HC.png) | No | 29 weeks 1 day / Good / Cryptic: Consistent with a cryptic pregnancy | 29 weeks 1 day / Good / Cryptic: Consistent with a cryptic pregnancy |
| hc18_standin_testset_000_HC (000_HC.png) | Not sure | 29 weeks 1 day / Good / Possibly cryptic: Possibly cryptic: answer the question above to confirm | 29 weeks 1 day / Good / Possibly cryptic: Possibly cryptic: answer the question above to confirm |
| abdomen_cryptic_before (Patient00773_Plane2_1_of_1.png) | Yes | 23 weeks 0 days / Limited / Not cryptic: May be not cryptic: the pregnancy was known | No estimate / - / Cannot assess: Cannot assess |
| abdomen_cryptic_before (Patient00773_Plane2_1_of_1.png) | No | 23 weeks 0 days / Limited / Cryptic: May be consistent with a cryptic pregnancy | No estimate / - / Cannot assess: Cannot assess |
| abdomen_cryptic_before (Patient00773_Plane2_1_of_1.png) | Not sure | 23 weeks 0 days / Limited / Possibly cryptic: May be cryptic: answer the question above to confirm | No estimate / - / Cannot assess: Cannot assess |
| abdomen_poor_showed_age (Patient01500_Plane2_1_of_1.png) | Yes | 21 weeks 5 days / Poor / Cannot assess: Cannot assess | No estimate / - / Cannot assess: Cannot assess |
| abdomen_poor_showed_age (Patient01500_Plane2_1_of_1.png) | No | 21 weeks 5 days / Poor / Cannot assess: Cannot assess | No estimate / - / Cannot assess: Cannot assess |
| abdomen_poor_showed_age (Patient01500_Plane2_1_of_1.png) | Not sure | 21 weeks 5 days / Poor / Cannot assess: Cannot assess | No estimate / - / Cannot assess: Cannot assess |
| femur (Patient01288_Plane5_1_of_1.png) | Yes | 21 weeks 6 days / Limited / Not cryptic: Borderline around 20 weeks: it could fall on either side | No estimate / - / Cannot assess: Cannot assess |
| femur (Patient01288_Plane5_1_of_1.png) | No | 21 weeks 6 days / Limited / Possibly cryptic: Borderline around 20 weeks: it could fall on either side | No estimate / - / Cannot assess: Cannot assess |
| femur (Patient01288_Plane5_1_of_1.png) | Not sure | 21 weeks 6 days / Limited / Possibly cryptic: Borderline around 20 weeks: it could fall on either side | No estimate / - / Cannot assess: Cannot assess |
| thorax (Patient01063_Plane6_1_of_1.png) | Yes | 18 weeks 4 days / Limited / Not cryptic: Borderline around 20 weeks: it could fall on either side | No estimate / - / Cannot assess: Cannot assess |
| thorax (Patient01063_Plane6_1_of_1.png) | No | 18 weeks 4 days / Limited / Possibly cryptic: Borderline around 20 weeks: it could fall on either side | No estimate / - / Cannot assess: Cannot assess |
| thorax (Patient01063_Plane6_1_of_1.png) | Not sure | 18 weeks 4 days / Limited / Possibly cryptic: Borderline around 20 weeks: it could fall on either side | No estimate / - / Cannot assess: Cannot assess |
| trans_thalamic_refused_by_current_check (Patient01322_Plane3_3_of_4.png) | Yes | No estimate / - / Cannot assess: Cannot assess | No estimate / - / Cannot assess: Cannot assess |
| trans_thalamic_refused_by_current_check (Patient01322_Plane3_3_of_4.png) | No | No estimate / - / Cannot assess: Cannot assess | No estimate / - / Cannot assess: Cannot assess |
| trans_thalamic_refused_by_current_check (Patient01322_Plane3_3_of_4.png) | Not sure | No estimate / - / Cannot assess: Cannot assess | No estimate / - / Cannot assess: Cannot assess |
| trans_thalamic_accepted (Patient01056_Plane3_1_of_2.png) | Yes | 36 weeks 1 day / Good / Not cryptic: Not cryptic: the pregnancy was known | 36 weeks 1 day / Good / Not cryptic: Not cryptic: the pregnancy was known |
| trans_thalamic_accepted (Patient01056_Plane3_1_of_2.png) | No | 36 weeks 1 day / Good / Cryptic: Consistent with a cryptic pregnancy | 36 weeks 1 day / Good / Cryptic: Consistent with a cryptic pregnancy |
| trans_thalamic_accepted (Patient01056_Plane3_1_of_2.png) | Not sure | 36 weeks 1 day / Good / Possibly cryptic: Possibly cryptic: answer the question above to confirm | 36 weeks 1 day / Good / Possibly cryptic: Possibly cryptic: answer the question above to confirm |

## Poor / first-trimester cases

| Case | Answer | Before: age shown / image check / result | After: age shown / image check / result |
|---|---|---|---|
| fixture_crl_week12 (crl_week12.png) | No | No estimate / - / Cannot assess: Cannot assess | No estimate / - / Cannot assess: Cannot assess |
| fixture_week12_user_crop (week12_user_crop.png) | No | 14 weeks 0 days / Poor / Cannot assess: Cannot assess | No estimate / - / Cannot assess: Cannot assess |
| poor_head_passes_head_check (Patient01426_Plane3_1_of_4.png) | Yes | 20 weeks 0 days / Poor / Cannot assess: Cannot assess | No estimate / - / Cannot assess: Cannot assess |
| poor_head_passes_head_check (Patient01426_Plane3_1_of_4.png) | No | 20 weeks 0 days / Poor / Cannot assess: Cannot assess | No estimate / - / Cannot assess: Cannot assess |
| poor_head_passes_head_check (Patient01426_Plane3_1_of_4.png) | Not sure | 20 weeks 0 days / Poor / Cannot assess: Cannot assess | No estimate / - / Cannot assess: Cannot assess |

Reasons shown after the change on the No estimate card: head check refusal = "This image does not look like a standard fetal head view, so no estimate is given."; Poor image = "The image check is Poor, so an age estimate would be unreliable. Why: ...".
