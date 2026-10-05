"""ResNet50 transfer-learning model (stage classifier + head-circumference regressor) and Grad-CAM."""
import cv2
import keras
import numpy as np
import tensorflow as tf

from .config import DROPOUT, HEAD_UNITS, IMG_SIZE, LAST_CONV_LAYER


def build_backbone(weights="imagenet") -> keras.Model:
    """Frozen ResNet50 that maps a 224x224x3 image to a 2048-d embedding."""
    base = keras.applications.ResNet50(include_top=False, weights=weights,
                                       input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(base.output)
    return keras.Model(base.input, x, name="resnet50_features")


def _head(x, n_classes: int):
    x = keras.layers.Dense(HEAD_UNITS, activation="relu", name="head_dense")(x)
    x = keras.layers.Dropout(DROPOUT, name="head_dropout")(x)
    return keras.layers.Dense(n_classes, activation="softmax", name="head_out")(x)


def _reg_head(x):
    x = keras.layers.Dense(HEAD_UNITS, activation="relu", name="reg_dense")(x)
    x = keras.layers.Dropout(DROPOUT, name="reg_dropout")(x)
    return keras.layers.Dense(1, name="reg_out")(x)


def build_head(n_classes: int, feat_dim: int = 2048) -> keras.Model:
    """Stage classifier trained on cached bottleneck features."""
    inp = keras.Input((feat_dim,), name="features")
    return keras.Model(inp, _head(inp, n_classes), name="head")


def build_reg_head(feat_dim: int = 2048) -> keras.Model:
    """Head-circumference regressor (standardised target) on cached features."""
    inp = keras.Input((feat_dim,), name="features")
    return keras.Model(inp, _reg_head(inp), name="reg_head")


def build_locator(map_shape=(7, 7, 2048)) -> keras.Model:
    """Head localiser: 1x1 convolutions on the ResNet50 conv5 map -> per-cell head logit.
    Trained on HC18 annotation ellipses; used to find the skull for the Grad-CAM check."""
    inp = keras.Input(map_shape, name="conv_map")
    x = keras.layers.Conv2D(128, 1, activation="relu", name="loc_hidden")(inp)
    return keras.Model(inp, keras.layers.Conv2D(1, 1, name="loc_out")(x), name="locator")


def assemble(backbone: keras.Model, head: keras.Model, reg_head: keras.Model) -> keras.Model:
    """Backbone + both trained heads as one flat model with outputs [stage probs, HC z-score].

    Flat so Grad-CAM can reach the conv layers. The HC output is standardised; the app
    maps it back to mm with the mean/std stored in metadata.json.
    """
    feats = backbone.output
    full = keras.Model(backbone.input, [_head(feats, head.output.shape[-1]), _reg_head(feats)],
                       name="hc18_stage_and_age")
    for name, src in (("head_dense", head), ("head_out", head),
                      ("reg_dense", reg_head), ("reg_out", reg_head)):
        full.get_layer(name).set_weights(src.get_layer(name).get_weights())
    return full


def build_grad_model(model: keras.Model) -> keras.Model:
    return keras.Model(model.inputs, [model.get_layer(LAST_CONV_LAYER).output, *model.outputs])


def grad_cam(model: keras.Model, x: np.ndarray, grad_model: keras.Model | None = None):
    """Return (heatmap in [0,1], stage probabilities, HC z-score) for one preprocessed image.

    The heatmap explains the head-circumference regression output, which is the basis
    of the gestational-age estimate.
    """
    grad_model = grad_model or build_grad_model(model)
    xb = tf.convert_to_tensor(x[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv, probs, z = grad_model(xb, training=False)
        score = z[:, 0]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return cam, probs[0].numpy(), float(z[0, 0])


def overlay(gray: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a Grad-CAM heatmap over a grayscale image. Returns RGB uint8."""
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(cv2.addWeighted(heat, alpha, base, 1 - alpha, 0), cv2.COLOR_BGR2RGB)


class Analyzer:
    """One compiled forward/backward pass that returns the embedding, class probabilities,
    HC z-score and Grad-CAM heatmap. Eager ResNet50 calls are very slow on CPU, so the graph
    is traced once and reused."""

    def __init__(self, model: keras.Model, locator: keras.Model | None = None):
        grad_model = build_grad_model(model)

        @tf.function
        def run(xb):
            with tf.GradientTape() as tape:
                conv, probs, z = grad_model(xb, training=False)
                score = z[:, 0]
            loc = tf.sigmoid(locator(conv, training=False)) if locator is not None else tf.zeros_like(conv[..., :1])
            return conv, probs, z, tape.gradient(score, conv), loc   # sum over the two copies

        self._run = run

    def __call__(self, x: np.ndarray):
        # Batch of one takes ~19 s on this TensorFlow CPU build (batch >= 2 takes ~0.2 s), so the
        # image is duplicated and only the first copy is used. Samples are independent at inference.
        xb = tf.convert_to_tensor(np.stack([x, x]), dtype=tf.float32)
        conv, probs, z, grads, loc = self._run(xb)
        weights = tf.reduce_mean(grads[:1], axis=(0, 1, 2))
        cam = tf.nn.relu(tf.reduce_sum(conv[0] * weights, axis=-1)).numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        embedding = conv.numpy()[0].mean(axis=(0, 1))      # = the avg_pool features
        head_prob = cv2.resize(loc[0, ..., 0].numpy(), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        return cam, probs[0].numpy(), float(z[0, 0]), embedding, head_prob
