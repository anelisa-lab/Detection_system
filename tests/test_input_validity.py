"""Input-validity tests: the app must give an age for head views and refuse everything else.

Needs a trained model in artifacts/ (python train.py ...). Run: python -m pytest tests -q
"""
import io
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import pytest

from hcml.growth import CRL_RANGE_MM, robinson_ga_weeks

ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "artifacts"))

needs_model = pytest.mark.skipif(not (MODEL_DIR / "ood_reference.npz").exists(),
                                 reason="train a model first (python train.py)")


@pytest.fixture(scope="module")
def predictor():
    from hcml.pipeline import Predictor
    return Predictor(MODEL_DIR)


def _dicom_bytes(gray: np.ndarray) -> bytes:
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.Rows, ds.Columns = "US", *gray.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit, ds.PixelRepresentation = 7, 0
    ds.PixelData = gray.astype(np.uint8).tobytes()
    ds.is_little_endian, ds.is_implicit_VR = True, False
    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds)
    return buf.getvalue()


def _photo_jpeg() -> bytes:
    """A colourful, non-ultrasound picture: smooth gradients with a few shapes."""
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:480, 0:640]
    img = np.stack([(x / 640 * 255), (y / 480 * 255), ((x + y) / 1120 * 255)], -1).astype(np.uint8)
    for _ in range(12):
        cv2.circle(img, (int(rng.integers(640)), int(rng.integers(480))), int(rng.integers(20, 90)),
                   [int(c) for c in rng.integers(0, 255, 3)], -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


@needs_model
@pytest.mark.parametrize("name", ["valid_head_a.png", "valid_head_b.png"])
def test_valid_head_view_gets_an_estimate(predictor, name):
    r = predictor.predict((FIX / name).read_bytes(), name)
    assert r.level in ("good", "reduced")
    assert r.ga_weeks is not None and 11 <= r.ga_weeks <= 41
    assert r.half_days >= predictor.reg["interval_days"]       # never narrower than the validation error


@needs_model
def test_week12_crl_view_is_refused(predictor):
    r = predictor.predict((FIX / "crl_week12.png").read_bytes(), "crl_week12.png")
    assert r.level == "rejected" and r.ga_weeks is None and r.cam is None
    # refused by the head-view check or, failing that, the image check; either way with a plain reason
    assert ("does not look like a standard fetal head view" in r.notes[0]
            or "does not look like a standard head circumference view" in r.notes[0])


@needs_model
def test_non_ultrasound_photo_is_refused(predictor):
    r = predictor.predict(_photo_jpeg(), "holiday.jpg")
    assert r.level == "rejected" and r.ga_weeks is None


@needs_model
def test_dicom_head_view_is_accepted_and_dicom_crl_is_refused(predictor):
    head = cv2.imread(str(FIX / "valid_head_a.png"), cv2.IMREAD_GRAYSCALE)
    crl = cv2.imread(str(FIX / "crl_week12.png"), cv2.IMREAD_GRAYSCALE)
    ok = predictor.predict(_dicom_bytes(head), "head.dcm")
    bad = predictor.predict(_dicom_bytes(crl), "crl.dcm")
    assert ok.ga_weeks is not None and ok.level in ("good", "reduced")
    assert bad.ga_weeks is None and bad.level == "rejected"


@needs_model
def test_unreadable_and_empty_files_raise_clear_errors(predictor):
    with pytest.raises(ValueError):
        predictor.predict(b"", "empty.png")
    with pytest.raises(Exception):
        predictor.predict(b"not an image at all", "junk.png")


@needs_model
def test_false_rejection_rate_on_valid_scans_is_low(predictor):
    files = sorted((ROOT.parent / "data" / "test_set").glob("*.png"))[:60]
    if not files:
        pytest.skip("HC18 test_set not available")
    rejected = sum(predictor.predict(p.read_bytes(), p.name).level == "rejected" for p in files)
    assert rejected / len(files) <= 0.05


def test_robinson_crl_formula():
    # Robinson & Fleming: 8.052*sqrt(CRL)+23.73 days; CRL 60 mm is about 12 weeks 2 days
    assert abs(robinson_ga_weeks(60) * 7 - (8.052 * 60 ** 0.5 + 23.73)) < 1e-6
    assert 12.0 <= robinson_ga_weeks(60) <= 12.5
    assert CRL_RANGE_MM == (10.0, 84.0)


def test_scan_summary_wording_by_stage():
    from hcml.content import what_scan_suggests

    late = " ".join(what_scan_suggests(24.0))
    early = " ".join(what_scan_suggests(15.0))
    assert "20 weeks or more" in late and "far along" in late and "second trimester" in late
    assert "earlier stage" in early and "Typical at this stage" in early
    for t in (late, early):
        assert "diagnos" not in t.lower().replace("not a diagnostic", "") and "you have a cryptic" not in t.lower()
