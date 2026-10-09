"""Unit tests for the cryptic pregnancy screening verdict (no model needed)."""
import pytest

from hcml.screening import (CANNOT, CRYPTIC, NO, NO_FETUS, NOT_CRYPTIC, NOT_SURE, POSSIBLY, YES, screen)

HALF = 14.0   # +/- 14 days = +/- 2 weeks


@pytest.mark.parametrize("badge", ["Poor", "Rejected"])
def test_poor_image_cannot_assess(badge):
    s = screen(25.0, HALF, badge, NO)
    assert s.badge == CANNOT and s.label == "Cannot assess" and "not a reliable head view" in s.why


def test_withheld_estimate_cannot_assess():
    s = screen(None, HALF, "Good", NO)
    assert s.badge == CANNOT and "not a reliable head view" in s.why


@pytest.mark.parametrize("answer", [YES, NO, NOT_SURE])
def test_under_20_weeks_is_not_cryptic_whatever_the_answer(answer):
    s = screen(15.0, HALF, "Good", answer)
    assert s.badge == NOT_CRYPTIC and s.label == "Not cryptic by the usual definition"
    assert not s.borderline and "15 weeks" in s.why


def test_late_and_no_is_consistent_with_cryptic():
    s = screen(28.0, HALF, "Good", NO)
    assert s.badge == CRYPTIC and s.label == "Consistent with a cryptic pregnancy"


def test_late_and_yes_is_known():
    s = screen(28.0, HALF, "Good", YES)
    assert s.badge == NOT_CRYPTIC and s.label == "Not cryptic: the pregnancy was known"


def test_late_and_not_sure_is_possibly_cryptic():
    s = screen(28.0, HALF, "Good", NOT_SURE)
    assert s.badge == POSSIBLY and s.label == "Possibly cryptic: answer the question above to confirm"


def test_exactly_20_weeks_counts_as_late():
    assert screen(20.0 + HALF / 7 + 0.1, HALF, "Good", NO).badge == CRYPTIC


@pytest.mark.parametrize("ga", [18.5, 20.0, 21.5])
def test_range_straddling_20_weeks_is_borderline_and_shows_both_readings(ga):
    s = screen(ga, HALF, "Good", NO)
    assert s.borderline and "Borderline" in s.label
    labels = [l for _, l in s.readings]
    assert labels == ["Not cryptic by the usual definition", "Consistent with a cryptic pregnancy"]
    assert s.badge == POSSIBLY


def test_borderline_with_yes_is_not_cryptic_on_both_sides():
    s = screen(20.0, HALF, "Good", YES)
    assert s.borderline and s.badge == NOT_CRYPTIC and len(s.readings) == 2


def test_range_just_clear_of_20_is_not_borderline():
    assert not screen(22.1, HALF, "Good", NO).borderline       # low end 20.1
    assert not screen(17.9, HALF, "Good", NO).borderline       # high end 19.9


@pytest.mark.parametrize("answer,expect", [
    (NO, "May be consistent with a cryptic pregnancy"),
    (YES, "May be not cryptic: the pregnancy was known"),
    (NOT_SURE, "May be cryptic: answer the question above to confirm"),
])
def test_limited_image_softens_every_late_verdict(answer, expect):
    s = screen(30.0, HALF, "Limited", answer, image_reason="the heatmap is not on the skull")
    assert s.softened and s.label == expect
    assert any("the heatmap is not on the skull" in n and "'may be'" in n for n in s.notes)


def test_limited_image_softens_early_and_borderline():
    assert screen(15.0, HALF, "Limited", NO).label == "May be not cryptic by the usual definition"
    b = screen(20.0, HALF, "Limited", NO)
    assert b.borderline and [l for _, l in b.readings][0].startswith("May be")


def test_found_out_at_20_weeks_or_later_is_cryptic():
    s = screen(28.0, HALF, "Good", YES, weeks_found=22)
    assert s.badge == CRYPTIC and s.label.startswith("Cryptic by the usual definition")
    assert "found out at about 22 weeks" in s.why
    assert screen(28.0, HALF, "Good", YES, weeks_found=20).badge == CRYPTIC


def test_found_out_before_20_weeks_stays_known():
    s = screen(28.0, HALF, "Good", YES, weeks_found=12)
    assert s.badge == NOT_CRYPTIC and s.label == "Not cryptic: the pregnancy was known" and not s.notes
    assert screen(28.0, HALF, "Good", YES).badge == NOT_CRYPTIC


def test_found_out_late_is_softened_when_image_is_limited():
    s = screen(28.0, HALF, "Limited", YES, weeks_found=22, image_reason="the heatmap is not on the skull")
    assert s.badge == CRYPTIC and s.label.startswith("May be cryptic by the usual definition")


def test_found_out_late_is_cryptic_even_when_scan_is_under_20_weeks():
    s = screen(15.0, HALF, "Good", YES, weeks_found=24)
    assert s.badge == CRYPTIC and s.label.startswith("Cryptic by the usual definition")
    assert not s.borderline and any("do not agree" in n for n in s.notes)


def test_found_out_late_settles_a_borderline_estimate():
    s = screen(20.0, HALF, "Good", YES, weeks_found=22)
    assert s.badge == CRYPTIC and not s.borderline


def test_cannot_assess_tells_the_user_to_see_a_clinician():
    assert "see a clinician regardless" in screen(None, HALF, "Good", NO).sentence


def test_weeks_found_later_than_scan_note():
    assert any("later than the scan" in n for n in screen(22.0, HALF, "Good", YES, weeks_found=34).notes)


def test_unknown_answer_defaults_to_not_sure():
    assert screen(28.0, HALF, "Good", "maybe").badge == POSSIBLY


def test_pdf_report_contains_verdict_answer_and_reason():
    import numpy as np
    from hcml.report import build_report

    s = screen(28.0, HALF, "Good", NO)
    gray = (np.random.default_rng(0).random((224, 224)) * 255).astype("uint8")
    pdf = build_report([{"name": "a.png", "gray": gray, "heat": None, "headline": "28 weeks 0 days",
                         "range_text": None, "trimester_text": None, "due_text": None,
                         "screen_badge": s.badge, "screen_label": s.label,
                         "screen_lines": [s.sentence, s.why, "Your answer: No"], "summary": ["x"]}])
    assert pdf[:4] == b"%PDF" and len(pdf) > 2000


@pytest.mark.parametrize("answer", [YES, NO, NOT_SURE])
def test_no_fetal_head_gives_no_fetus_seen_whatever_the_answer(answer):
    s = screen(None, HALF, "No fetus", answer, 22.0 if answer == YES else None)
    assert s.badge == NO_FETUS and s.label == "No fetal head seen in this scan"
    assert "cannot rule a pregnancy out" in s.sentence and "pregnancy test" in s.sentence
    assert not s.borderline and not s.readings and s.kind == "grey"
