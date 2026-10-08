RESULTS
=======
RESULTS_SUMMARY.md        one-page summary with before/after numbers and 95% confidence ranges
tables/                   the same numbers as CSV (percentages with ci_low / ci_high):
  barcelona_test_share_accepted.csv      by plane: existing check, Barcelona-only (before), shipped 95% and 98% (after)
  hc18_heads_share_accepted.csv          HC18 held-out 218 and test_set 335, overall and by age band (held-out bands use Hadlock age
                                          from head circumference; test_set bands use the app's estimated age)
  combined_with_existing_check.csv       non-head / head images that still get an age or a label when both checks must accept
  operating_points_validation.csv        thresholds and validation acceptance at 90 / 95 / 98%
  app_before_after_cases.csv, user_1_2HC_before_after.csv   what the real app showed before and after the change
safety_failures/          non-head images that got an age and a Cryptic / Possibly cryptic label:
  before_change_existing_check_alone_barcelona_test_split.csv: 109 rows
  before_change_old_app_1741_image_sample.csv: 52 rows
  remaining_after_change_95pct_barcelona_test_split.csv: 2 rows
  remaining_after_change_98pct_barcelona_test_split.csv: 7 rows
example_images/           false accepts, false refusals (Barcelona), HC18 heads still refused; study1_barcelona_only/ = the earlier version
app_screenshots/          real app screenshots before and after (abdomen, femur, thorax, trans-thalamic, 1_2HC.png with Yes / No / Not sure)
Operating point: 95% accepts 95% of validation heads (app default); 98% accepts 98%. "Existing image check" = the embedding-distance
check that was already in the app. Images are from FETAL_PLANES_DB and HC18 (CC BY 4.0); see 05_share_with_group/NOTICE.md.
