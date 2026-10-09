"""Train the 'is a fetal head present?' check (artifacts/presence.npz).

    python scripts/train_presence.py                       # head-free proxies made from tests/fixtures
    python scripts/train_presence.py --erase-from data/hc18/training_set --max-erase 150
    python scripts/train_presence.py --negatives path/to/non_pregnant_scans    # real negatives (best)

Negatives are ultrasound images with no fetal head:
  --negatives   a folder of real scans of people who are not pregnant (PNG/JPEG/BMP/DICOM). Use these if
                you can: proxies are only a stand-in.
  --erase-from  a folder of head-circumference scans; the head found by the locator is painted over with
                background-like speckle or inpainted, leaving the fan, texture and gain of a real scan.
Positives are the training embeddings already stored in artifacts/ood_reference.npz.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from hcml.pipeline import Predictor
from hcml.preprocess import clean, read_gray, to_model_input
from hcml.presence import fit_presence

EXT = {".png", ".jpg", ".jpeg", ".bmp", ".dcm", ".dicom"}


def files(folder, pattern="*"):
    return sorted(p for p in Path(folder).rglob(pattern) if p.suffix.lower() in EXT and "_Annotation" not in p.name)


def embed(pred, gray):
    return pred.analyzer(to_model_input(clean(gray, denoise=pred.meta.get("denoise", True))))[3].reshape(-1)


def erase_head(pred, gray, mode, rng):
    """Paint over the head the locator finds, keeping the rest of the scan. Returns None if no head found."""
    h, w = gray.shape
    prob = pred.analyzer(to_model_input(clean(gray, denoise=pred.meta.get("denoise", True))))[4]
    m = cv2.resize((prob >= 0.5).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    if m.sum() < 0.02 * m.size:
        return None
    m = cv2.dilate(m, np.ones((max(5, min(h, w) // 20) | 1,) * 2, np.uint8))
    if mode == "inpaint":
        return cv2.inpaint(gray, m * 255, 15, cv2.INPAINT_TELEA)
    bg = gray[m == 0]
    mu, sd = float(bg.mean()), float(bg.std())
    fill = cv2.GaussianBlur(np.clip(rng.normal(mu * rng.uniform(0.4, 0.9), sd * rng.uniform(0.4, 0.8), gray.shape), 0, 255)
                            .astype(np.float32), (0, 0), rng.uniform(0.8, 2.0))
    return np.clip(np.where(m > 0, fill, gray), 0, 255).astype(np.uint8)


def variants(gray, rng):
    yield gray
    yield cv2.flip(gray, 1)
    yield np.clip(gray.astype(np.float32) * rng.uniform(0.7, 1.4), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="artifacts")
    ap.add_argument("--negatives", help="folder of real scans with no fetal head")
    ap.add_argument("--erase-from", default=str(Path(__file__).resolve().parent.parent / "tests" / "fixtures"),
                    help="folder of head-circumference scans to erase the head from (default: tests/fixtures)")
    ap.add_argument("--max-erase", type=int, default=200, help="most source scans to erase heads from")
    ap.add_argument("--seeds", type=int, default=4, help="speckle variants per erased scan")
    ap.add_argument("--target-pass", type=float, default=0.98, help="share of real head scans that must pass")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = Path(a.model_dir)
    pred = Predictor(d)
    rng = np.random.default_rng(a.seed)
    pos = np.load(d / "ood_reference.npz")["feats"].astype(np.float32)

    neg, n_real, n_proxy = [], 0, 0
    if a.negatives:
        for f in files(a.negatives):
            try:
                g = read_gray(str(f), f.name)
            except Exception as e:           # noqa: BLE001 - skip unreadable files, keep going
                print("skip", f.name, e)
                continue
            for v in variants(g, rng):
                neg.append(embed(pred, v))
                n_real += 1
    src = [f for f in files(a.erase_from) if "head" in f.name.lower() and "nonhead" not in f.name.lower()
           and "no_fetus" not in f.name.lower()]
    rng.shuffle(src)
    for f in src[: a.max_erase]:
        g = read_gray(str(f), f.name)
        for mode in ("speckle", "inpaint"):
            for _ in range(a.seeds if mode == "speckle" else 1):
                e = erase_head(pred, g, mode, rng)
                if e is None:
                    continue
                for v in variants(e, rng):
                    neg.append(embed(pred, v))
                    n_proxy += 1
    if not neg:
        sys.exit("no negatives found")
    model, info = fit_presence(pos, np.array(neg), a.target_pass, seed=a.seed)
    info.update(real_negative_rows=n_real, proxy_negative_rows=n_proxy)
    model.save(d / "presence.npz")
    meta_path = d / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["presence"] = info
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
