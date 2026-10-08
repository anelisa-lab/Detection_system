"""Embeddings (and optional conv5 maps) from the committed model's frozen encoder, for the head-view check study.

The encoder is the ResNet50 inside artifacts/model.keras up to `conv5_block3_out`, with the same preprocessing the
app uses (resize, denoise, 3-channel, ResNet50 normalisation). The embedding is the spatial mean of that map, which
is exactly what `Predictor` feeds to the current image check. Nothing is trained and nothing in artifacts/ changes.

    python scripts/head_check_extract.py --list images.csv --out cache/fetal [--maps]

`images.csv` needs the columns `image` and `path`. Writes `<out>_emb.npy` (N x 2048 float32), `<out>_index.csv`, and
with --maps `<out>_maps.npy` (N x 7 x 7 x 2048 float16).
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import keras
import numpy as np
import pandas as pd

from hcml import config as C
from hcml.preprocess import clean, read_gray, to_model_input


def load_encoder(model_dir: str):
    model = keras.models.load_model(Path(model_dir) / "model.keras", compile=False)
    meta = json.loads((Path(model_dir) / "metadata.json").read_text())
    enc = keras.Model(model.input, model.get_layer(C.LAST_CONV_LAYER).output)
    return enc, bool(meta.get("denoise", True))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True, help="prefix for the output files (outside the repo for big arrays)")
    ap.add_argument("--model-dir", default="artifacts")
    ap.add_argument("--maps", action="store_true", help="also keep the 7x7x2048 conv5 maps (float16)")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    if Path(args.out).resolve().parent == Path(args.model_dir).resolve():
        sys.exit("--out must not be inside the model folder")

    items = pd.read_csv(args.list)
    enc, denoise = load_encoder(args.model_dir)
    n = len(items)
    emb = np.zeros((n, 2048), np.float32)
    maps = None
    if args.maps:
        maps = np.lib.format.open_memmap(f"{args.out}_maps.npy", mode="w+", dtype=np.float16, shape=(n, 7, 7, 2048))
    for start in range(0, n, args.batch):
        rows = items.iloc[start:start + args.batch]
        x = np.stack([to_model_input(clean(read_gray(Path(p).read_bytes(), Path(p).name), denoise=denoise))
                      for p in rows["path"]])
        m = enc.predict(x, verbose=0)
        emb[start:start + len(rows)] = m.mean(axis=(1, 2))
        if maps is not None:
            maps[start:start + len(rows)] = m.astype(np.float16)
        if (start // args.batch) % 20 == 0:
            print(f"  {min(start + args.batch, n)}/{n}", flush=True)
    np.save(f"{args.out}_emb.npy", emb)
    items.to_csv(f"{args.out}_index.csv", index=False)
    if maps is not None:
        maps.flush()
    print("done ->", args.out)


if __name__ == "__main__":
    main()
