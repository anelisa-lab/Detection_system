"""Head circumference <-> gestational age, trimesters and due dates.

Formula: Hadlock FP, Deter RL, Harrist RB, Park SK. "Estimating fetal age:
computer-assisted analysis of multiple fetal growth measurements."
Radiology 1984;152:497-501.   GA(weeks) = 8.96 + 0.540*HC + 0.0003*HC^3, HC in cm.
It is a population mean curve, valid for roughly 12-40 weeks.
"""
from datetime import date, timedelta

import numpy as np

FORMULA_CITATION = "Hadlock et al., Radiology 1984;152:497-501 (head-circumference formula)"


def hadlock_ga_weeks(hc_mm):
    hc_cm = np.asarray(hc_mm, dtype=np.float64) / 10.0
    return 8.96 + 0.540 * hc_cm + 0.0003 * hc_cm ** 3


def hadlock_hc_mm(ga_weeks):
    """Inverse of the Hadlock curve (it is monotonic), by interpolation."""
    grid = np.linspace(20.0, 400.0, 3801)
    return np.interp(ga_weeks, hadlock_ga_weeks(grid), grid)


def weeks_days(ga_weeks: float) -> tuple[int, int]:
    total = int(round(ga_weeks * 7))
    return total // 7, total % 7


def fmt_weeks_days(ga_weeks: float) -> str:
    w, d = weeks_days(ga_weeks)
    return f"{w} weeks {d} day{'s' if d != 1 else ''}"


def trimester(ga_weeks: float) -> int:
    w, _ = weeks_days(ga_weeks)
    return 1 if w < 14 else (2 if w < 28 else 3)


def due_date(ga_weeks: float, today: date | None = None) -> date:
    """Estimated due date: today plus the time left to 40 weeks."""
    return (today or date.today()) + timedelta(days=round((40.0 - ga_weeks) * 7))


# First trimester: crown-rump length (CRL) instead of head circumference.
CRL_CITATION = "Robinson & Fleming, Br J Obstet Gynaecol 1975;82:702-710"
CRL_RANGE_MM = (10.0, 84.0)      # about 7 to 14 weeks


def robinson_ga_weeks(crl_mm: float) -> float:
    """Gestational age from CRL: GA(days) = 8.052*sqrt(CRL mm) + 23.73."""
    return float((8.052 * np.sqrt(crl_mm) + 23.73) / 7.0)
