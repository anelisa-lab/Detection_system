"""App-level tests of the workstation UI, run through Streamlit's AppTest (needs a trained model)."""
import os
import re
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest

from hcml.content import UNRELIABLE_SENTENCE
from hcml.pipeline import LIMITED_WARNING

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures"
HARNESS = str(Path(__file__).parent / "app_with_upload.py")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "artifacts"))
needs_model = pytest.mark.skipif(not (MODEL_DIR / "ood_reference.npz").exists(), reason="train a model first")

GOOD, LATE, EARLY = FIX / "valid_head_good.png", FIX / "valid_head_late.png", FIX / "valid_head_early.png"
NEAR20 = FIX / "valid_head_b.png"      # estimate close to 20 weeks, so its range straddles 20
WEEK12, CRL = FIX / "week12_user_crop.png", FIX / "crl_week12.png"
NO_FETUS_SCAN = FIX / "no_fetus_a.png"


def run(paths, answer="Not sure", nav=None):
    from streamlit.testing.v1 import AppTest

    paths = paths if isinstance(paths, (list, tuple)) else [paths]
    os.environ["UPLOAD_FILES"] = "|".join(str(p) for p in paths)
    at = AppTest.from_file(HARNESS, default_timeout=600).run()
    if answer != "Not sure":
        question(at).set_value(answer).run()
    if nav:
        at.radio(key="nav").set_value(nav).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def question(at):
    return next(r for r in at.radio if r.label.startswith("Did you know"))


def html_of(at):
    return " ".join(m.value for m in at.markdown)


def image_badge(at):
    m = re.search(r'<span class="chip-k">Image check</span><span class="badge (\w+)"><svg.*?<span>(\w+)</span>', html_of(at), re.S)
    return m.group(2) if m else None


def verdict_text(at):
    m = re.search(r'Cryptic pregnancy screening result.*?<span class="badge big \w+">.*?<span>([^<]+)</span>.*?v-title">([^<]+)<',
                  html_of(at), re.S)
    return (m.group(1), m.group(2)) if m else None


@needs_model
def test_only_the_one_question_remains():
    at = run(GOOD)
    assert question(at).value == "Not sure"
    labels = " ".join(w.label for w in list(at.selectbox) + list(at.number_input) + list(at.checkbox)
                      + list(at.date_input) + list(at.text_input)).lower()
    for word in ("period", "pregnancy test", "bmi", "weight", "height", "symptom"):
        assert word not in labels
    question(at).set_value("Yes").run()
    assert any("find out" in w.label for w in at.number_input)


@needs_model
def test_good_image_shows_summary_and_stage_guide():
    at = run(LATE)      # a head view the image check rates Good
    assert image_badge(at) == "Good"
    h = html_of(at)
    assert "What this scan suggests" in h and "The scan suggests a gestational age of about" in h
    assert "About this stage" in h
    assert UNRELIABLE_SENTENCE not in h


@needs_model
def test_poor_image_shows_no_age_and_replaces_summary():
    """A Poor image gives the verdict Cannot assess, so the page shows No estimate instead of an age."""
    at = run(WEEK12)
    h = html_of(at)
    assert "No estimate" in h and 'class="hero num"' not in h
    assert not any(LIMITED_WARNING in w.value for w in at.warning)      # there is no estimate to call rough
    assert UNRELIABLE_SENTENCE in h and "The scan suggests a gestational age" not in h
    assert "The image check is Poor" in h and "Why:" in h


@needs_model
def test_limited_image_still_shows_its_age_with_the_warning():
    at = run(GOOD)                      # image check Limited: the estimate is shown, flagged as rough
    assert image_badge(at) == "Limited"
    assert any(LIMITED_WARNING in w.value for w in at.warning)
    assert 'class="hero num"' in html_of(at)


@needs_model
def test_late_stage_verdicts_follow_the_answer():
    assert verdict_text(run(LATE, "No"))[1] == "Consistent with a cryptic pregnancy"
    assert verdict_text(run(LATE, "Yes"))[1] == "Not cryptic: the pregnancy was known"
    at = run(LATE)
    assert verdict_text(at)[0] == "Possibly cryptic"
    assert "promptly for confirmation, dating and antenatal care" in html_of(at)      # supportive action


@needs_model
def test_early_stage_is_not_cryptic():
    at = run(EARLY, "No")
    assert verdict_text(at) == ("Not cryptic", "Not cryptic by the usual definition")
    assert "promptly for confirmation, dating and antenatal care" not in html_of(at)


@needs_model
def test_bad_images_cannot_assess():
    for f in (CRL, WEEK12):
        at = run(f, "No")
        badge, label = verdict_text(at)
        assert badge == "Cannot assess" and label == "Cannot assess"
        assert "not a reliable head view" in html_of(at)


@needs_model
def test_scan_with_no_fetus_says_so_and_gives_no_age():
    if not (MODEL_DIR / "presence.npz").exists():
        pytest.skip("this model folder has no presence.npz")
    for answer in ("Not sure", "No"):
        at = run(NO_FETUS_SCAN, answer)
        assert verdict_text(at) == ("No fetus seen", "No fetal head seen in this scan")
        h = html_of(at)
        assert "No fetal head was found" in h and "Not pregnant" in h
        assert "cannot rule a pregnancy out" in h
        assert "Cryptic by the usual definition" not in h and "Consistent with a cryptic" not in h
        assert "promptly for confirmation, dating and antenatal care" not in h
        assert image_badge(at) is None or image_badge(at) == "No fetus"


@needs_model
def test_borderline_scan_shows_both_readings():
    # The wording may be softened ("May be consistent ...") if a retrain turns this scan Limited.
    h = html_of(run(NEAR20, "No"))
    assert "Borderline around 20 weeks" in h and "onsistent with a cryptic pregnancy" in h


@needs_model
def test_workstation_chrome_and_accessibility():
    at = run(LATE)
    h = html_of(at)
    for part in ('class="timeline"', 'class="tiles"', 'class="ws-footer"', 'class="viewer-bar"'):
        assert part in h
    assert "not validated on cryptic pregnancies" in h and "clinician confirms everything" in h
    imgs = re.findall(r"<img [^>]*>", h)
    assert imgs and all('alt="' in i and 'alt=""' not in i for i in imgs)
    for badge in re.findall(r'<span class="badge [^"]*">(.*?)</span></span>', h, re.S):
        assert "<svg" in badge                        # status is never colour alone: icon + text
    assert len(at.tabs) >= 3 and [r.label for r in at.radio if r.label == "Navigation"]
    names = re.findall(r"<span>(Good|Limited|Poor|Rejected|No fetus)</span>", h)
    assert names


@needs_model
def test_light_theme_toggle_swaps_palette():
    at = run(GOOD)
    assert "--bg:#0E1116" in html_of(at)
    at.toggle(key="light").set_value(True).run()
    assert "--bg:#F3F6FA" in html_of(at) and "--bg:#0E1116" not in html_of(at)


@needs_model
def test_compare_toggle_shows_side_by_side():
    at = run(LATE)
    assert "canvas compare" not in html_of(at)
    at.toggle(key="compare").set_value(True).run()
    assert "canvas compare" in html_of(at)


@needs_model
def test_batch_table_filter_sort_and_open():
    files = sorted((ROOT.parent / "data" / "test_set").glob("*.png"))[:12]
    if len(files) < 12:
        pytest.skip("HC18 test_set not available")
    at = run(files)
    assert at.radio(key="nav").value == "Batch"                   # many files start on the table
    h = html_of(at)
    assert 'class="strip"' in h and "Scans" in h
    assert h.count('class="cell-name"') == 12
    assert len([b for b in at.button if b.key and b.key.startswith("open_")]) == 12
    # filter to one badge: fewer rows, all of that badge
    at.multiselect(key="f_badge").set_value(["Good"]).run()
    rows = html_of(at).count('class="cell-name"')
    assert 0 < rows < 12
    # sort and open a scan: jumps to the viewer on that scan
    at.selectbox(key="sort").set_value("Estimate, high to low").run()
    first_key = next(b.key for b in at.button if b.key and b.key.startswith("open_"))
    at.button(key=first_key).click().run()
    assert at.radio(key="nav").value == "Scan" and f"{files[int(first_key.split('_')[1])].name}" in html_of(at)


@needs_model
def test_unreadable_file_in_a_batch_is_reported_not_fatal(tmp_path):
    bad = tmp_path / "junk.png"
    bad.write_bytes(b"not an image")
    at = run([GOOD, bad, LATE, EARLY, CRL, WEEK12])        # 6 files -> batch view
    assert "Unreadable" in html_of(at)
