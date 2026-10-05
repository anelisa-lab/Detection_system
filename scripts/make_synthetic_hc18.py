"""Write a small fake dataset in HC18's layout, for smoke-testing the code only.

The images are noisy ellipses, not ultrasound scans. Never report results from them.

    python scripts/make_synthetic_hc18.py --out data/synthetic/training_set --n 120
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=120)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = ["filename,pixel size(mm),head circumference (mm)"]
    for i in range(args.n):
        h, w = 540, 800
        img = rng.normal(40, 15, (h, w)).clip(0, 255).astype(np.uint8)
        hc_mm = float(rng.uniform(60, 340))
        px = float(rng.uniform(0.12, 0.18))
        ratio = rng.uniform(0.7, 0.9)
        perim_px = hc_mm / px
        a = perim_px / (math.pi * (1 + ratio) * 1.0)  # rough semi-axis from circumference
        a = min(a, 250.0)
        b = a * ratio
        centre = (int(rng.uniform(300, 500)), int(rng.uniform(220, 320)))
        ang = float(rng.uniform(0, 180))
        cv2.ellipse(img, (centre, (int(2 * a), int(2 * b)), ang), 200, 6)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        ann = np.zeros((h, w), np.uint8)
        cv2.ellipse(ann, (centre, (int(2 * a), int(2 * b)), ang), 255, 1)
        name = f"{i:03d}_HC.png"
        cv2.imwrite(str(out / name), img)
        cv2.imwrite(str(out / name.replace(".png", "_Annotation.png")), ann)
        rows.append(f"{name},{px:.6f},{hc_mm:.2f}")
    (out / "training_set_pixel_size_and_HC.csv").write_text("\n".join(rows) + "\n")
    print(f"wrote {args.n} synthetic images to {out}")


if __name__ == "__main__":
    main()
