"""Train and evaluate the HC18 gestational-stage model described in the GROUP11 guide.

    python train.py --data data/hc18/training_set --out artifacts

Pipeline: preprocess (resize, denoise, 3-channel) -> frozen ResNet50 features ->
two 256-unit dense heads (head-circumference regression, mapped to gestational age
with the Hadlock formula, and the early/mid/late classifier) with early stopping -> one evaluation on the held-out test
split -> Grad-CAM examples -> saved model the web app loads.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import keras
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hcml import config as C
from hcml.data import add_labels, annotation_features, annotation_mask, load_hc18
from hcml.growth import FORMULA_CITATION, hadlock_ga_weeks
from hcml.model import Analyzer, assemble, build_backbone, build_head, build_locator, build_reg_head, grad_cam, overlay
from hcml.preprocess import clean, read_gray, to_model_input
from hcml.skull import cam_overlap, ellipse_mask, peak_inside, skull_ellipse
from hcml.splitting import grouped_split, screening_20w
from hcml.validity import fit_reference


def split(df: pd.DataFrame):
    """Patient-grouped 80/20 split, then 20% of train for validation (see hcml/splitting.py)."""
    return grouped_split(df, C.TEST_FRACTION, C.VAL_FRACTION, C.SEED)


def extract(df: pd.DataFrame, backbone, denoise: bool, batch: int = 32):
    """Return (2048-d features, cleaned 224x224 grayscale images, conv5 maps) for every row.
    `backbone` outputs the 7x7x2048 conv5 map; the pooled feature is its spatial mean
    (identical to the avg_pool layer)."""
    feats, grays, maps = [], [], []
    for start in range(0, len(df), batch):
        rows = df.iloc[start:start + batch]
        g = [clean(read_gray(p), denoise=denoise) for p in rows["image_path"]]
        x = np.stack([to_model_input(i) for i in g])
        mp = backbone.predict(x, verbose=0)
        feats.append(mp.mean(axis=(1, 2)))
        maps.append(mp.astype(np.float16))
        grays.extend(g)
        print(f"  features {min(start + batch, len(df))}/{len(df)}", end="\r")
    print()
    return np.concatenate(feats), np.stack(grays), np.concatenate(maps)


def specificity_per_class(cm: np.ndarray) -> np.ndarray:
    total = cm.sum()
    spec = []
    for k in range(cm.shape[0]):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        tn = total - tp - fp - fn
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)
    return np.array(spec)


def evaluate(y_true, y_pred, names) -> dict:
    labels = list(range(len(names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    spec = specificity_per_class(cm)
    w = s / s.sum()
    weighted_spec = float((spec * w).sum())
    return {
        "test_size": int(s.sum()),
        "weighted": {
            "recall_sensitivity": float((r * w).sum()),   # support-weighted recall == accuracy; not screening recall
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float((p * w).sum()),
            "f1": float((f * w).sum()),
            "specificity": weighted_spec,
            "false_positive_rate": 1 - weighted_spec,
        },
        "per_class": {n: {"precision": float(p[i]), "recall_sensitivity": float(r[i]),
                          "f1": float(f[i]), "specificity": float(spec[i]), "support": int(s[i])}
                      for i, n in enumerate(names)},
        "confusion_matrix": cm.tolist(),
    }


def evaluate_regression(hc_true, hc_pred) -> dict:
    """Errors in HC (mm) and in gestational age (days, via the Hadlock curve)."""
    ga_true, ga_pred = hadlock_ga_weeks(hc_true), hadlock_ga_weeks(hc_pred)
    err_days = np.abs(ga_pred - ga_true) * 7
    ss_res = float(((ga_true - ga_pred) ** 2).sum())
    ss_tot = float(((ga_true - ga_true.mean()) ** 2).sum())
    return {
        "n": int(len(hc_true)),
        "mae_days": float(err_days.mean()),
        "median_abs_error_days": float(np.median(err_days)),
        "rmse_days": float(np.sqrt(((ga_pred - ga_true) ** 2).mean()) * 7),
        "mae_hc_mm": float(np.abs(hc_pred - hc_true).mean()),
        "r2_ga": 1 - ss_res / ss_tot if ss_tot else 0.0,
        "within_7_days": float((err_days <= 7).mean()),
        "within_14_days": float((err_days <= 14).mean()),
    }


def plot_regression(hc_true, hc_pred, path):
    ga_true, ga_pred = hadlock_ga_weeks(hc_true), hadlock_ga_weeks(hc_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(ga_true, ga_pred, s=10, alpha=0.6)
    lim = [min(ga_true.min(), ga_pred.min()), max(ga_true.max(), ga_pred.max())]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("True gestational age (weeks, from HC)")
    ax.set_ylabel("Predicted (weeks)")
    ax.set_title("Test-set age regression")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, names, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i][j], ha="center", va="center",
                    color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    ax.set_title("Test-set confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(hist, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, key in zip(axes, ("loss", "accuracy")):
        ax.plot(hist[key], label="train")
        ax.plot(hist[f"val_{key}"], label="validation")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gradcam(model, grays, y_true, y_pred, names, path, rng):
    right = np.flatnonzero(y_true == y_pred)
    wrong = np.flatnonzero(y_true != y_pred)
    picks = [("correct", i) for i in rng.choice(right, min(2, len(right)), replace=False)]
    picks += [("misclassified", i) for i in rng.choice(wrong, min(2, len(wrong)), replace=False)]
    if not picks:
        return
    fig, axes = plt.subplots(1, len(picks), figsize=(3.2 * len(picks), 3.6))
    for ax, (kind, i) in zip(np.atleast_1d(axes), picks):
        cam, _, _ = grad_cam(model, to_model_input(grays[i]))
        ax.imshow(overlay(grays[i], cam))
        ax.set_title(f"{kind}\ntrue {names[y_true[i]]} / pred {names[y_pred[i]]}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="HC18 training_set folder (PNG images + annotations)")
    ap.add_argument("--out", default="artifacts", help="where the model and reports go")
    ap.add_argument("--thresholds", type=float, nargs=2, default=C.HC_THRESHOLDS_MM,
                    metavar=("EARLY_MID_MM", "MID_LATE_MM"), help="HC cut points in mm")
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--epochs", type=int, default=C.MAX_EPOCHS)
    ap.add_argument("--weights", default="imagenet", help="'imagenet' or a local .h5 weights file")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(C.SEED)
    names = C.CLASS_NAMES

    df = add_labels(load_hc18(args.data), args.thresholds)
    print(f"{len(df)} images; class counts:", df["stage"].value_counts().reindex(names).to_dict())
    df, split_info = split(df)
    print(f"{split_info['n_patients']} patients; patients per split:", split_info["patients_per_split"])
    print(df.groupby(["stage", "split"]).size().unstack(fill_value=0).reindex(names))

    print("Annotation features (descriptive only, not model inputs)...")
    feats_tab = pd.DataFrame([annotation_features(r.image_path, r.annotation_path, r.pixel_mm, r.hc_mm)
                              for r in df.itertuples()])
    pd.concat([df[["filename", "stage", "split", "ga_weeks_hadlock", "pixel_mm"]], feats_tab], axis=1) \
        .to_csv(out / "reports" / "dataset_features.csv", index=False)

    print("Extracting frozen ResNet50 features...")
    backbone = build_backbone(args.weights)
    maps_model = keras.Model(backbone.input, backbone.get_layer(C.LAST_CONV_LAYER).output)
    X, grays, maps = extract(df, maps_model, denoise=not args.no_denoise)
    y = df["label"].to_numpy()
    m = {s: (df["split"] == s).to_numpy() for s in ("train", "val", "test")}

    # Input-validity reference: train+val embeddings, cut points calibrated on valid scans never
    # used for fitting (the test split, plus the unlabeled HC18 test_set folder when present).
    held = [X[m["test"]]]
    ext = Path(args.data).parent / "test_set"
    if ext.is_dir():
        ext_df = pd.DataFrame({"image_path": sorted(str(p) for p in ext.glob("*.png"))})
        if len(ext_df):
            print(f"Embedding {len(ext_df)} extra valid scans from {ext} for validity calibration...")
            held.append(extract(ext_df, maps_model, denoise=not args.no_denoise)[0])
    ref, validity_info = fit_reference(X[~m["test"]], np.concatenate(held))
    ref.save(out / "ood_reference.npz")
    print("Validity thresholds:", json.dumps(validity_info))

    # Head localiser on the conv5 map, trained on HC18 annotation ellipses (for the Grad-CAM check).
    cells = maps.shape[1]
    gt224 = np.stack([annotation_mask(p, C.IMG_SIZE) for p in df["annotation_path"]])
    gt_cells = np.stack([cv2.resize(g, (cells, cells), interpolation=cv2.INTER_AREA) for g in gt224])[..., None]
    locator = build_locator(maps.shape[1:])
    locator.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.BinaryCrossentropy(from_logits=True))
    locator.fit(maps[m["train"]].astype(np.float32), gt_cells[m["train"]],
                validation_data=(maps[m["val"]].astype(np.float32), gt_cells[m["val"]]),
                epochs=args.epochs, batch_size=C.BATCH_SIZE, verbose=2,
                callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE,
                                                         restore_best_weights=True)])
    locator.save(out / "locator.keras")
    prob_test = 1 / (1 + np.exp(-locator.predict(maps[m["test"]].astype(np.float32), verbose=0)[..., 0]))
    ious = []
    for pr, gt in zip(prob_test, gt224[m["test"]]):
        pm = ellipse_mask(skull_ellipse(cv2.resize(pr, (C.IMG_SIZE, C.IMG_SIZE), interpolation=cv2.INTER_LINEAR)))
        gm = (gt >= 0.5).astype(np.uint8)
        ious.append((pm & gm).sum() / max((pm | gm).sum(), 1))
    locator_info = {"test_mean_iou": float(np.mean(ious)), "test_iou_ge_0.5": float(np.mean(np.array(ious) >= 0.5))}
    print("Skull locator (test):", json.dumps(locator_info))
    del maps

    head = build_head(len(names), X.shape[1])
    head.compile(optimizer=keras.optimizers.Adam(C.LEARNING_RATE),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = head.fit(X[m["train"]], y[m["train"]], validation_data=(X[m["val"]], y[m["val"]]),
                    epochs=args.epochs, batch_size=C.BATCH_SIZE, verbose=2,
                    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE,
                                                             restore_best_weights=True)])
    plot_history(hist.history, out / "reports" / "training_curves.png")

    # Regression head: standardised head circumference (mm), same splits and early stopping.
    hc = df["hc_mm"].to_numpy(dtype=np.float64)
    hc_mean, hc_std = float(hc[m["train"]].mean()), float(hc[m["train"]].std())
    z = ((hc - hc_mean) / hc_std).astype(np.float32)
    reg = build_reg_head(X.shape[1])
    reg.compile(optimizer=keras.optimizers.Adam(C.LEARNING_RATE), loss="mse", metrics=["mae"])
    rhist = reg.fit(X[m["train"]], z[m["train"]], validation_data=(X[m["val"]], z[m["val"]]),
                    epochs=args.epochs, batch_size=C.BATCH_SIZE, verbose=2,
                    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE,
                                                             restore_best_weights=True)])
    to_mm = lambda zz: zz.reshape(-1) * hc_std + hc_mean
    # Uncertainty band shown in the app: 80th percentile of validation errors, in days.
    val_err = np.abs(hadlock_ga_weeks(to_mm(reg.predict(X[m["val"]], verbose=0)))
                     - hadlock_ga_weeks(hc[m["val"]])) * 7
    interval_days = float(np.percentile(val_err, 80))

    # The test split is used once, here, after everything above is fixed.
    y_test = y[m["test"]]
    y_pred = head.predict(X[m["test"]], verbose=0).argmax(1)
    metrics = evaluate(y_test, y_pred, names)
    metrics["epochs_trained"] = len(hist.history["loss"])
    hc_pred = to_mm(reg.predict(X[m["test"]], verbose=0))
    metrics["regression"] = evaluate_regression(hc[m["test"]], hc_pred)
    metrics["regression"]["epochs_trained"] = len(rhist.history["loss"])
    metrics["regression"]["interval_days_80pct_validation"] = interval_days
    print("Age regression (test):", json.dumps(metrics["regression"], indent=2))
    metrics["screening_20w"] = screening_20w(hc[m["test"]], hc_pred, groups=df.loc[m["test"], "patient_id"].to_numpy())
    metrics["split"] = split_info
    print("20-week screening (test):", json.dumps(metrics["screening_20w"], indent=2))
    plot_regression(hc[m["test"]], hc_pred, out / "reports" / "regression_scatter.png")
    print(classification_report(y_test, y_pred, labels=list(range(len(names))), target_names=names, zero_division=0))
    print(json.dumps(metrics["weighted"], indent=2))
    plot_confusion(metrics["confusion_matrix"], names, out / "reports" / "confusion_matrix.png")
    (out / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    full = assemble(backbone, head, reg)
    full.save(out / "model.keras")

    # How often does Grad-CAM land on the skull for valid test scans? (sets expectations for the badge)
    analyzer = Analyzer(full, locator)
    ov, pk = [], []
    for g in grays[m["test"]]:
        cam, _, _, _, hp = analyzer(to_model_input(g))
        mk = ellipse_mask(skull_ellipse(hp))
        ov.append(cam_overlap(cam, mk))
        pk.append(peak_inside(cam, mk))
    heat_info = {"test_median_overlap": float(np.median(ov)), "test_share_overlap_ge_50": float(np.mean(np.array(ov) >= 0.5)),
                 "test_share_peak_on_skull": float(np.mean(pk))}
    print("Grad-CAM on skull (valid test scans):", json.dumps(heat_info))
    ga_train = hadlock_ga_weeks(hc[~m["test"]])
    (out / "metadata.json").write_text(json.dumps({
        "class_names": names,
        "hc_thresholds_mm": list(args.thresholds),
        "img_size": C.IMG_SIZE,
        "denoise": not args.no_denoise,
        "task": "Gestational age regression (head circumference mapped with Hadlock 1984) plus a 3-class stage classifier (proxy task, guide Section 3.4)",
        "metrics": metrics["weighted"],
        "validity": validity_info,
        "locator": locator_info,
        "heatmap_check": {**heat_info, "ga_p5_weeks": float(np.percentile(ga_train, 5)),
                          "min_overlap": 0.5},
        "regression": {
            "hc_mean_mm": hc_mean, "hc_std_mm": hc_std,
            "formula": FORMULA_CITATION,
            "interval_days": interval_days,
            "test": metrics["regression"],
        },
        "per_class": metrics["per_class"],
        "screening_20w": metrics["screening_20w"],
        "split": split_info,
    }, indent=2))
    df.drop(columns=["image_path", "annotation_path"]).to_csv(out / "reports" / "splits.csv", index=False)

    print("Grad-CAM examples...")
    plot_gradcam(full, grays[m["test"]], y_test, y_pred, names,
                 out / "reports" / "gradcam_examples.png", np.random.default_rng(C.SEED))
    print(f"Done. Model: {out / 'model.keras'}  Reports: {out / 'reports'}")


if __name__ == "__main__":
    main()
