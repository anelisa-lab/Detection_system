"""Rewrite the three number tables in README.md from the result CSVs, so README and 03_results cannot disagree.

    python scripts/update_readme_numbers.py [--readme README.md] [--tables project_package/03_results/tables]

The tables are replaced in place (found by their header line); no number is typed by hand. Source CSVs are written by
scripts/package_build.py from eval_head_check/test_report.json and eval_head_check_hc18/test_report.json.
"""
import argparse
import re
from pathlib import Path

import pandas as pd


def f(r, p):
    return f"{r[p + '_pct']:.1f}% ({r[p + '_ci_low']:.1f}-{r[p + '_ci_high']:.1f})"


def replace_table(text, header_start, new_table):
    lines = text.split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith(header_start))
    j = i
    while j < len(lines) and lines[j].startswith("|"):
        j += 1
    return "\n".join(lines[:i] + new_table.split("\n") + lines[j:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--tables", default="project_package/03_results/tables")
    a = ap.parse_args()
    t = Path(a.tables)
    text = Path(a.readme).read_text(encoding="utf8")

    bar = pd.read_csv(t / "barcelona_test_share_accepted.csv").set_index("group")
    rows = [("Head: all brain", "Heads, all brain planes"), ("Head: trans-thalamic (HC plane)", "Trans-thalamic heads"),
            ("NON-HEAD: all (incl. Other)", "Non-head, all (incl. Other)"), ("NON-HEAD: without Other", "Non-head, without Other"),
            ("Non-head: Fetal abdomen", "Abdomen"), ("Non-head: Fetal femur", "Femur"), ("Non-head: Fetal thorax", "Thorax"),
            ("Non-head: Maternal cervix", "Maternal cervix"), ("Non-head: Other", "Other")]
    tab = ["| Share accepted | Existing image check | Study 1: Barcelona-only, 95% / 98% | **Shipped (study 2): with HC18 heads, 95%** | **Shipped: 98%** |",
           "|---|---|---|---|---|"]
    for g, lab in rows:
        r = bar.loc[g]
        tab.append(f"| {lab} | {f(r, 'existing_image_check')} | {f(r, 'barcelona_only_95')} / {f(r, 'barcelona_only_98')} | "
                   f"{f(r, 'shipped_95')} | {f(r, 'shipped_98')} |")
    text = replace_table(text, "| Share accepted | Existing image check | Study 1: Barcelona-only", "\n".join(tab))

    hc = pd.read_csv(t / "hc18_heads_share_accepted.csv")
    def hrow(setname, band, lab):
        r = hc[(hc["set"] == setname) & (hc["age_band"] == band)].iloc[0]
        return (f"| {lab} | {f(r, 'existing_image_check')} | {f(r, 'barcelona_only_95')} / {f(r, 'barcelona_only_98')} | "
                f"{f(r, 'shipped_95')} | {f(r, 'shipped_98')} |"), r
    tab = ["| Share accepted | Existing image check | Study 1 at 95% / 98% | **Shipped, 95%** | **Shipped, 98%** |", "|---|---|---|---|---|"]
    for setname, band, lab in [("HC18 held-out 218", "all", "Held-out 218, all"),
                               ("HC18 held-out 218", "under 17 weeks", "... under 17 weeks (n=61)"),
                               ("HC18 held-out 218", "17 to 20 weeks", "... 17 to 20 weeks (n=49)"),
                               ("HC18 held-out 218", "20 weeks or more", "... 20 weeks or more (n=108)"),
                               ("HC18 unlabeled test_set 335", "all", "Unlabeled test_set 335, all")]:
        tab.append(hrow(setname, band, lab)[0])
    text = replace_table(text, "| Share accepted | Existing image check | Study 1 at 95%", "\n".join(tab))

    cb = pd.read_csv(t / "combined_with_existing_check.csv")
    c95 = cb[cb["operating_point"] == 95].set_index("group")
    c98 = cb[cb["operating_point"] == 98].set_index("group")
    tab = ["| | Existing check alone | Combined, 95% | Combined, 98% |", "|---|---|---|---|"]
    n = int(c95.loc["non-head, all", "n"])
    tab.append(f"| Non-head images that still get an age (n={n:,}) | {f(c95.loc['non-head, all'], 'current_alone_gets_age')} | "
               f"{f(c95.loc['non-head, all'], 'combined_gets_age')} | {f(c98.loc['non-head, all'], 'combined_gets_age')} |")
    tab.append(f"| ... and a Cryptic or Possibly cryptic label | {f(c95.loc['non-head, all'], 'current_alone_gets_cryptic_or_possibly')} | "
               f"{f(c95.loc['non-head, all'], 'combined_gets_cryptic_or_possibly')} | {f(c98.loc['non-head, all'], 'combined_gets_cryptic_or_possibly')} |")
    tab.append(f"| Barcelona heads that get an age (n={int(c95.loc['heads', 'n'])}) | {f(c95.loc['heads'], 'current_alone_gets_age')} | "
               f"{f(c95.loc['heads'], 'combined_gets_age')} | {f(c98.loc['heads'], 'combined_gets_age')} |")
    for setname, band, lab in [("HC18 held-out 218", "all", "HC18 held-out heads that get an age (n=218)"),
                               ("HC18 unlabeled test_set 335", "all", "HC18 test_set heads that get an age (n=335)")]:
        r = hc[(hc["set"] == setname) & (hc["age_band"] == band)].iloc[0]
        tab.append(f"| {lab} | {f(r, 'existing_image_check')} | {f(r, 'shipped_95_AND_existing')} | {f(r, 'shipped_98_AND_existing')} |")
    text = replace_table(text, "| | Existing check alone | Combined, 95%", "\n".join(tab))
    k = lambda g, p, c: int(c.loc[g, p + "_k"])
    note = (f"Counts: of {n:,} non-head images, {k('non-head, all', 'combined_gets_age', c95)} get an age and "
            f"{k('non-head, all', 'combined_gets_cryptic_or_possibly', c95)} get a label at 95% "
            f"({k('non-head, all', 'combined_gets_age', c98)} and {k('non-head, all', 'combined_gets_cryptic_or_possibly', c98)} at 98%), "
            f"against {k('non-head, all', 'current_alone_gets_age', c95)} and {k('non-head, all', 'current_alone_gets_cryptic_or_possibly', c95)} with the existing check alone.")
    text = re.sub(r"\nCounts: of [\d,]+ non-head images.*\n", "\n", text)
    text = text.replace("\n".join(tab), "\n".join(tab) + "\n\n" + note, 1)
    Path(a.readme).write_text(text, encoding="utf8")
    print("README tables regenerated from", t)


if __name__ == "__main__":
    main()
