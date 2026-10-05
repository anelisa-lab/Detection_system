"""Skull localisation helpers: ellipse from a predicted head-probability map, Grad-CAM overlap,
and detection of solid black boxes (redaction blocks) in the uploaded image."""
import cv2
import numpy as np

from .config import IMG_SIZE


def skull_ellipse(prob: np.ndarray, thr: float = 0.5):
    """Fit an ellipse to the largest blob of prob >= thr. Returns a cv2 ellipse or None."""
    m = (prob >= thr).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if len(c) < 5 or cv2.contourArea(c) < 0.02 * prob.size:
        return None
    return cv2.fitEllipse(c)


def ellipse_mask(ell, size: int = IMG_SIZE) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    if ell is not None:
        cv2.ellipse(m, ell, 1, -1)
    return m


def cam_overlap(cam: np.ndarray, mask: np.ndarray) -> float:
    """Share of the heatmap's total weight that lies inside the skull mask."""
    total = float(cam.sum())
    return float((cam * mask).sum() / total) if total > 0 else 0.0


def peak_inside(cam: np.ndarray, mask: np.ndarray) -> bool:
    y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return bool(mask[y, x])


def black_boxes(raw_small: np.ndarray, min_frac: float = 0.01) -> list[tuple[int, int, int, int]]:
    """Solid black rectangles (x, y, w, h), e.g. redaction blocks over text. Expects the image
    resized WITHOUT denoising so blocked pixels are still exactly dark."""
    z = (raw_small <= 2).astype(np.uint8)
    z = cv2.morphologyEx(z, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(z)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area >= min_frac * z.size and min(w, h) >= 8 and area / float(w * h) >= 0.9:
            boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def boxes_cover(mask: np.ndarray, boxes) -> float:
    """Fraction of the skull mask covered by black boxes."""
    if mask.sum() == 0 or not boxes:
        return 0.0
    cover = np.zeros_like(mask)
    for x, y, w, h in boxes:
        cover[y:y + h, x:x + w] = 1
    return float((cover * mask).sum() / mask.sum())
