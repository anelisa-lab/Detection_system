"""Regression tests for the head-view check (hcml/headcheck.py) and for the Cannot assess display fix.

Fixtures: HC18-style head views (valid_head_*), the two first-trimester images (crl_week12, week12_user_crop) and
FETAL_PLANES_DB images (see fixtures/NOTICE_FETAL_PLANES.md). Needs the committed model in artifacts/.
Run: python -m pytest tests -q
"""
import os
import re
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pytest

from hcml.headcheck import ENV_VAR, NOT_HEAD_VIEW, HeadCheck
from hcml.screening import CANNOT, screen

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "artifacts"))
HEAD_DIR = MODEL_DIR / "head_check"

needs_model = pytest.mark.skipif(not (MODEL_DIR / "ood_reference.npz").exists(), reason="train a model first")
needs_head = pytest.mark.skipif(not (HEAD_DIR / "classifier.npz").exists(), reason="head_check files not installed")

HEADS = ["valid_head_good", "valid_head_late", "valid_head_early", "valid_head_a", "valid_head_b"]
NON_HEADS = ["nonhead_abdomen", "nonhead_femur", "nonhead_thorax"]
HERO_AGE = re.compile(r'class="hero num"')


def png(name):
    return (FIX / f"{name}.png").read_bytes()


@pytest.fixture(autouse=True)
def default_operating_point(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)


@pytest.fixture(scope="module")
def predictor():
    from hcml.pipeline import Predictor
    return Predictor(MODEL_DIR)


# ---------------------------------------------------------------------------------------------------------
# the module and its own files
# ---------------------------------------------------------------------------------------------------------

@needs_head
def test_files_are_separate_from_the_existing_artifacts():
    assert HEAD_DIR.name == "head_check" and HEAD_DIR.parent == MODEL_DIR
    assert {p.name for p in HEAD_DIR.iterdir()} == {"classifier.npz", "thresholds.json"}
    assert (MODEL_DIR / "model.keras").exists() and (MODEL_DIR / "ood_reference.npz").exists()


@needs_head
def test_operating_points_default_95_and_order():
    h = HeadCheck.load(HEAD_DIR)
    assert h.targets == [90, 95, 98] and h.target() == 95
    # accepting more heads needs a lower score threshold
    assert h.threshold(98) < h.threshold(95) < h.threshold(90)
    assert h.threshold() == h.threshold(95)


@needs_head
def test_operating_point_is_a_setting_not_code(monkeypatch):
    h = HeadCheck.load(HEAD_DIR)
    monkeypatch.setenv(ENV_VAR, "98")
    assert h.target() == 98 and h.threshold() == h.threshold(98)
    assert h.target(95) == 95                                    # an explicit choice wins over the environment
    monkeypatch.setenv(ENV_VAR, "97")
    with pytest.raises(ValueError, match="not available"):
        h.target()
    monkeypatch.setenv(ENV_VAR, "ninety")
    with pytest.raises(ValueError, match="must be one of"):
        h.target()


@needs_head
def test_a_score_between_the_two_thresholds_is_refused_at_95_and_accepted_at_98():
    h = HeadCheck.load(HEAD_DIR)
    mid = (h.threshold(95) + h.threshold(98)) / 2
    # an embedding whose standardised value points along the weights, with the logit set to `mid`
    emb = h.mean + h.scale * h.coef * ((mid - h.intercept) / float(h.coef @ h.coef))
    assert h.score(emb) == pytest.approx(mid, abs=1e-3)
    assert not h.accepts(emb, 95) and h.accepts(emb, 98)


# ---------------------------------------------------------------------------------------------------------
# the pipeline: accepted only if BOTH checks accept
# ---------------------------------------------------------------------------------------------------------

@needs_model
@needs_head
def test_existing_thresholds_are_unchanged(predictor):
    assert predictor.ref.warn == pytest.approx(0.194, abs=1e-3)
    assert predictor.ref.reject == pytest.approx(0.225, abs=1e-3)


@needs_model
@needs_head
@pytest.mark.parametrize("name", HEADS + ["head_trans_thalamic"])
def test_head_views_are_still_accepted_and_given_an_age(predictor, name):
    r = predictor.predict(png(name), name + ".png")
    assert r.ga_weeks is not None and r.rejected_by is None
    assert r.head_score >= r.head_threshold


@needs_model
@needs_head
def test_trans_thalamic_scan_keeps_its_age_and_good_badge(predictor):
    r = predictor.predict(png("head_trans_thalamic"), "t.png")
    assert r.badge == "Good" and r.ga_weeks == pytest.approx(36.2, abs=0.3)


@needs_model
@needs_head
@pytest.mark.parametrize("name", NON_HEADS)
def test_non_head_planes_are_refused_with_no_age(predictor, name):
    r = predictor.predict(png(name), name + ".png")
    assert r.ga_weeks is None and r.cam is None and r.rejected_by == "head check"
    assert r.notes == [NOT_HEAD_VIEW] and r.badge == "Rejected"
    assert r.head_score < r.head_threshold


@needs_model
@needs_head
def test_without_the_head_check_the_old_behaviour_returns(predictor, monkeypatch):
    """Shows what the new check adds: the old image check alone gave the abdomen scan an age."""
    monkeypatch.setattr(predictor, "head", None)
    r = predictor.predict(png("nonhead_abdomen"), "a.png")
    assert r.ga_weeks is not None and r.head_score is None


@needs_model
@needs_head
def test_both_checks_must_accept(predictor, monkeypatch):
    good = png("valid_head_late")
    assert predictor.predict(good, "g.png").ga_weeks is not None
    with monkeypatch.context() as m:                                      # head check refuses, image check would accept
        m.setitem(predictor.head.thresholds, 95, 1e9)
        r = predictor.predict(good, "g.png")
        assert r.ga_weeks is None and r.rejected_by == "head check"
    with monkeypatch.context() as m:                                      # image check refuses, head check would accept
        m.setattr(predictor.ref, "reject", 0.01)
        r = predictor.predict(good, "g.png")
        assert r.ga_weeks is None and r.rejected_by == "image check" and r.head_score >= r.head_threshold


@needs_model
@needs_head
def test_operating_point_changes_who_gets_in(predictor):
    n = "head_accepted_only_at_98"
    r95, r98 = predictor.predict(png(n), n, 1.0, 95), predictor.predict(png(n), n, 1.0, 98)
    assert r95.ga_weeks is None and r95.rejected_by == "head check"
    assert r98.ga_weeks == pytest.approx(29.2, abs=0.3) and r98.badge == "Good"
    assert r95.head_threshold == pytest.approx(predictor.head.threshold(95))
    assert r98.head_threshold == pytest.approx(predictor.head.threshold(98))


@needs_model
@needs_head
def test_environment_setting_is_used_when_no_target_is_passed(predictor, monkeypatch):
    monkeypatch.setenv(ENV_VAR, "98")
    r = predictor.predict(png("head_accepted_only_at_98"), "x.png")
    assert r.ga_weeks is not None and r.head_threshold == pytest.approx(predictor.head.threshold(98))


@needs_model
@needs_head
def test_first_trimester_fixtures(predictor):
    crl = predictor.predict(png("crl_week12"), "c.png")
    assert crl.ga_weeks is None                                           # refused by the head check (and the image check)
    crop = predictor.predict(png("week12_user_crop"), "w.png")
    assert crop.rejected_by is None and crop.badge == "Poor"              # passes both checks, but the image check says Poor
    s = screen(crop.ga_weeks, crop.half_days, crop.badge, "No", None, "; ".join(crop.badge_reasons))
    assert s.badge == CANNOT and "see a clinician regardless" in s.sentence


# ---------------------------------------------------------------------------------------------------------
# what the page shows
# ---------------------------------------------------------------------------------------------------------

def _ui():
    import test_app_ui as T
    return T


@needs_model
@needs_head
@pytest.mark.parametrize("name", NON_HEADS)
@pytest.mark.parametrize("answer", ["Yes", "No", "Not sure"])
def test_page_for_a_non_head_plane_has_no_age_and_no_cryptic_label(name, answer):
    T = _ui()
    at = T.run(FIX / f"{name}.png", answer)
    h = T.html_of(at)
    assert "No estimate" in h and not HERO_AGE.search(h)
    assert "does not look like a standard fetal head view" in h
    badge, label = T.verdict_text(at)
    assert badge == "Cannot assess" and label == "Cannot assess"
    assert "see a clinician regardless" in h and "Cryptic" not in (badge or "")


@needs_model
@needs_head
@pytest.mark.parametrize("name", ["crl_week12", "week12_user_crop", "head_poor_image_check"])
def test_a_cannot_assess_verdict_never_shows_an_age_headline(name):
    """The README promise: a Poor or refused scan shows No estimate, not '14 weeks 0 days'."""
    T = _ui()
    at = T.run(FIX / f"{name}.png", "No")
    h = T.html_of(at)
    assert T.verdict_text(at)[0] == "Cannot assess"
    assert "No estimate" in h and not HERO_AGE.search(h)
    assert "Pregnancy timeline" not in h and "Trimester" not in h


@needs_model
@needs_head
def test_poor_image_page_explains_why_and_hides_age_numbers():
    T = _ui()
    h = T.html_of(T.run(FIX / "head_poor_image_check.png", "Yes"))
    assert "The image check is Poor" in h and "Why:" in h


@needs_model
@needs_head
@pytest.mark.parametrize("answer,badge,label", [
    ("Yes", "Not cryptic", "Not cryptic: the pregnancy was known"),
    ("No", "Cryptic", "Consistent with a cryptic pregnancy"),
    ("Not sure", "Possibly cryptic", "Possibly cryptic: answer the question above to confirm"),
])
def test_a_good_head_view_still_gets_its_age_and_the_same_three_labels(answer, badge, label):
    T = _ui()
    at = T.run(FIX / "valid_head_late.png", answer)
    h = T.html_of(at)
    assert T.image_badge(at) == "Good" and re.search(r'class="hero num"[^>]*>3\d weeks', h)
    assert T.verdict_text(at) == (badge, label)


@needs_model
@needs_head
def test_setting_in_the_app_changes_the_operating_point(monkeypatch):
    from streamlit.testing.v1 import AppTest
    T = _ui()
    monkeypatch.setenv("UPLOAD_FILES", str(FIX / "head_accepted_only_at_98.png"))
    at = AppTest.from_file(T.HARNESS, default_timeout=600).run()
    box = next(s for s in at.selectbox if s.label == "Head-view check")
    assert box.value == 95 and "No estimate" in T.html_of(at)
    box.set_value(98).run()
    assert not at.exception
    assert not any("No estimate" in m.value and "hero muted" in m.value for m in at.markdown)
    assert re.search(r'class="hero num"[^>]*>29 weeks', T.html_of(at))


@needs_model
@needs_head
def test_environment_variable_sets_the_default_in_the_app(monkeypatch):
    T = _ui()
    monkeypatch.setenv(ENV_VAR, "98")
    at = T.run(FIX / "head_accepted_only_at_98.png", "No")
    box = next(s for s in at.selectbox if s.label == "Head-view check")
    assert box.value == 98 and re.search(r'class="hero num"', T.html_of(at))


@needs_model
@needs_head
def test_hc18_test_set_heads_are_not_falsely_refused(predictor):
    """The app was built on HC18, so the combined checks must keep accepting its unlabeled test_set heads.

    Needs the HC18 test_set next to the repo (../data/test_set). Measured on all 335 scans at the 95% point: 2 refused
    (0.6%); the image check alone also refuses 2."""
    ts = Path(os.environ.get("HC18_TEST_SET", ROOT.parent / "data" / "test_set"))
    files = sorted(ts.glob("*.png"))
    if not files:
        pytest.skip(f"HC18 test_set not found at {ts}: download test_set.zip from https://zenodo.org/records/1327317 and unzip it there, or set HC18_TEST_SET (see README, Running the tests)")
    refused = [p.name for p in files if predictor.predict(p.read_bytes(), p.name).ga_weeks is None]
    assert len(refused) / len(files) <= 0.02, refused
