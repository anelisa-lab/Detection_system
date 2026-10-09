"""A scan with no fetus says 'Not pregnant' where it used to say 'No estimate'; other refusals keep 'No estimate'."""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "artifacts"))
needs_presence = pytest.mark.skipif(not (MODEL_DIR / "presence.npz").exists(), reason="no presence.npz in this model folder")


def _photo_jpeg() -> bytes:
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:480, 0:640]
    img = np.stack([(x / 640 * 255), (y / 480 * 255), ((x + y) / 1120 * 255)], -1).astype(np.uint8)
    for _ in range(12):
        cv2.circle(img, (int(rng.integers(640)), int(rng.integers(480))), int(rng.integers(20, 90)),
                   [int(c) for c in rng.integers(0, 255, 3)], -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_grey_images_are_scans_and_colourful_ones_are_not():
    from hcml.pipeline import looks_like_scan
    assert looks_like_scan((FIX / "valid_head_good.png").read_bytes())
    assert looks_like_scan((FIX / "no_fetus_a.png").read_bytes())
    assert not looks_like_scan(_photo_jpeg())
    assert looks_like_scan(b"\x00" * 10)             # not a picture OpenCV can read (e.g. DICOM): treated as a scan


@pytest.fixture(scope="module")
def predictor():
    from hcml.pipeline import Predictor
    return Predictor(MODEL_DIR)


@pytest.mark.skipif(not (MODEL_DIR / "ood_reference.npz").exists(), reason="train a model first")
@needs_presence
def test_a_photo_is_never_called_not_pregnant(predictor):
    r = predictor.predict(_photo_jpeg(), "holiday.jpg")
    assert r.ga_weeks is None and not r.no_fetus


@pytest.mark.skipif(not (MODEL_DIR / "ood_reference.npz").exists(), reason="train a model first")
@needs_presence
def test_scan_with_no_fetus_says_not_pregnant_instead_of_no_estimate():
    import test_app_ui as T
    at = T.run(FIX / "no_fetus_a.png", "No")
    h = T.html_of(at)
    assert 'class="hero">Not pregnant' in h and "No estimate" not in h
    assert T.verdict_text(at)[0] == "No fetus seen"          # the verdict card is unchanged
    assert "No fetal head was found" in h


@pytest.mark.skipif(not (MODEL_DIR / "ood_reference.npz").exists(), reason="train a model first")
def test_other_refusals_still_say_no_estimate():
    import test_app_ui as T
    h = T.html_of(T.run(FIX / "crl_week12.png", "No"))
    assert "No estimate" in h and "Not pregnant" not in h
