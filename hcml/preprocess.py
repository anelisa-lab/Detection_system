"""Image loading and preprocessing: resize, denoise, 3-channel, ResNet50 normalisation."""
import io
from pathlib import Path

import cv2
import numpy as np

from .config import IMG_SIZE


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def read_dicom(data) -> np.ndarray:
    """Return a grayscale uint8 image from a DICOM path or bytes."""
    import pydicom

    ds = pydicom.dcmread(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
    arr = ds.pixel_array
    if arr.ndim == 4:            # multi-frame colour: take the first frame
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = cv2.cvtColor(_to_uint8(arr[..., :3]), cv2.COLOR_RGB2GRAY)
    elif arr.ndim == 3:          # multi-frame grayscale
        arr = arr[0]
    return _to_uint8(arr)


def read_gray(source, filename: str = "") -> np.ndarray:
    """Read PNG/JPEG/BMP/DICOM from a path or raw bytes as grayscale uint8."""
    name = (filename or str(source if not isinstance(source, (bytes, bytearray)) else "")).lower()
    if name.endswith((".dcm", ".dicom")):
        return read_dicom(source)
    if isinstance(source, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(source, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:          # maybe a DICOM without an extension
            return read_dicom(bytes(source))
        return img
    img = cv2.imread(str(Path(source)), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {source}")
    return img


def clean(gray: np.ndarray, denoise: bool = True) -> np.ndarray:
    """Resize to IMG_SIZE x IMG_SIZE and reduce speckle noise. Returns uint8 grayscale."""
    img = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    if denoise:
        img = cv2.fastNlMeansDenoising(img, None, h=7, templateWindowSize=7, searchWindowSize=21)
    return img


def to_model_input(clean_gray: np.ndarray) -> np.ndarray:
    """Grayscale uint8 (H, W) -> ResNet50-normalised float32 (H, W, 3)."""
    from keras.applications.resnet50 import preprocess_input

    rgb = np.repeat(clean_gray[..., None], 3, axis=-1).astype(np.float32)
    return preprocess_input(rgb)


def prepare(source, filename: str = "", denoise: bool = True):
    """Full pipeline for one image. Returns (clean grayscale, model input)."""
    g = clean(read_gray(source, filename), denoise=denoise)
    return g, to_model_input(g)
