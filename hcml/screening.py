"""Cryptic pregnancy screening result from the scan estimate, the image check and one question.

The scan is the main evidence (how far along the pregnancy looks). It cannot show whether the
person knew about the pregnancy, so that comes from a single question. 'Cryptic' follows the usual
definition: a pregnancy not recognised until about 20 weeks or later. This is a screening aid for
a clinician to confirm, never a diagnosis.
"""
from dataclasses import dataclass, field

from .growth import fmt_weeks_days

THRESHOLD_WEEKS = 20.0
YES, NO, NOT_SURE = "Yes", "No", "Not sure"
ANSWERS = [YES, NO, NOT_SURE]

CRYPTIC, POSSIBLY, NOT_CRYPTIC, CANNOT = "Cryptic", "Possibly cryptic", "Not cryptic", "Cannot assess"
KIND = {CRYPTIC: "amber", POSSIBLY: "amber", NOT_CRYPTIC: "blue", CANNOT: "grey"}

_LABEL = {
    "early": "Not cryptic by the usual definition",
    "No": "Consistent with a cryptic pregnancy",
    "Yes": "Not cryptic: the pregnancy was known",
    "Not sure": "Possibly cryptic: answer the question above to confirm",
    "Late": "Cryptic by the usual definition: recognised at about 20 weeks or later",
}
_SOFT_LABEL = {
    "early": "May be not cryptic by the usual definition",
    "No": "May be consistent with a cryptic pregnancy",
    "Yes": "May be not cryptic: the pregnancy was known",
    "Not sure": "May be cryptic: answer the question above to confirm",
    "Late": "May be cryptic by the usual definition: recognised at about 20 weeks or later",
}
_BADGE = {"early": NOT_CRYPTIC, "No": CRYPTIC, "Yes": NOT_CRYPTIC, "Not sure": POSSIBLY, "Late": CRYPTIC}
_SENTENCE = {
    "early": "The scan suggests the pregnancy is under 20 weeks, so it does not meet the usual definition of "
             "a cryptic pregnancy (one that is not recognised until about 20 weeks or later).",
    "No": "The scan suggests the pregnancy is 20 weeks or more, and you did not know you were pregnant before "
          "it, which is consistent with a cryptic pregnancy.",
    "Yes": "The scan suggests the pregnancy is 20 weeks or more, but you knew about it before this scan, so it "
           "is not cryptic.",
    "Not sure": "The scan suggests the pregnancy is 20 weeks or more. Whether it is cryptic depends on whether "
                "you knew you were pregnant, so please answer the question above.",
    "Late": "The scan suggests the pregnancy is 20 weeks or more, and you only found out about it at about 20 "
            "weeks or later, which fits the usual definition of a cryptic pregnancy.",
}


@dataclass
class Screening:
    badge: str
    label: str
    sentence: str
    why: str
    kind: str
    borderline: bool = False
    softened: bool = False
    readings: list[tuple[str, str]] = field(default_factory=list)   # (condition, label) when borderline
    notes: list[str] = field(default_factory=list)


def _side(late: bool, answer: str) -> str:
    return answer if late else "early"


def screen(ga_weeks: float | None, half_days: float, image_badge: str, answer: str = NOT_SURE,
           weeks_found: float | None = None, image_reason: str = "") -> Screening:
    """image_badge is the image check: Good, Limited, Poor or Rejected. ga_weeks is None when withheld."""
    if answer not in ANSWERS:
        answer = NOT_SURE
    if ga_weeks is None or image_badge in ("Poor", "Rejected"):
        return Screening(CANNOT, "Cannot assess",
                         "The scan is not a reliable head view, so no screening result is given. Please retake "
                         "the scan or use a standard head circumference view.",
                         "Why: the scan is not a reliable head view.", KIND[CANNOT])

    low, high = ga_weeks - half_days / 7.0, ga_weeks + half_days / 7.0
    soft = image_badge == "Limited"
    labels = _SOFT_LABEL if soft else _LABEL
    est = f"{fmt_weeks_days(ga_weeks)} (range {fmt_weeks_days(max(low, 0))} to {fmt_weeks_days(high)})"
    ans = answer + (f", found out at about {weeks_found:.0f} weeks" if answer == YES and weeks_found else "")
    why = f"Why: stage {est}; your answer: {ans}; image check: {image_badge}."
    late_main = ga_weeks >= THRESHOLD_WEEKS
    # Found out at about 20 weeks or later is cryptic by the usual definition, whatever the "Yes".
    found_late = answer == YES and weeks_found is not None and weeks_found >= THRESHOLD_WEEKS
    eff = "Late" if found_late else answer
    notes = []
    if answer == YES and weeks_found:
        if found_late and not late_main:
            notes.append(f"You found out at about {weeks_found:.0f} weeks. The usual definition counts a pregnancy "
                         "recognised at about 20 weeks or later as cryptic, but the scan suggests a shorter "
                         "pregnancy, so please discuss this with your clinician.")
        if weeks_found > high + 1:
            notes.append("The week you found out is later than the scan suggests. Please check it with your "
                         "clinician.")

    main = _side(late_main, eff)
    borderline = low < THRESHOLD_WEEKS <= high
    if soft:
        notes.append(f"The image check is Limited ({image_reason or 'see the reason under the badge'}), so "
                     "treat this as a 'may be' reading.")

    if borderline:
        readings = [("If the pregnancy is under 20 weeks", labels["early"]),
                    ("If it is 20 weeks or more", labels[eff])]
        badges = {_BADGE["early"], _BADGE[eff]}
        badge = badges.pop() if len(badges) == 1 else POSSIBLY
        return Screening(badge, "Borderline around 20 weeks: it could fall on either side",
                         "The estimate is close to 20 weeks, so the result could fall on either side. Both "
                         "readings are shown.", why, KIND[badge], True, soft, readings, notes)
    return Screening(_BADGE[main], labels[main], _SENTENCE[main], why, KIND[_BADGE[main]], False, soft, [], notes)
