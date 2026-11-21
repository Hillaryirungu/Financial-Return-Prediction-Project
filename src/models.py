from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_mlp(lookback: int = 60) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(lookback,)),
        layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def build_cnn_gaf(lookback: int = 60) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(lookback, lookback, 2)),
        layers.Conv2D(16, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(2),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def build_lstm(lookback: int = 60, n_features: int = 1,
               n_outputs: int = 1, units: int = 32) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(units, dropout=0.1),
        layers.Dense(16 if n_outputs == 1 else 32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(n_outputs),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def gaf_images(X: "np.ndarray") -> "np.ndarray":
    """Convert (samples, time) to two-channel GASF/GADF images."""
    import numpy as np

    lo = X.min(axis=1, keepdims=True)
    hi = X.max(axis=1, keepdims=True)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    z = np.clip((2 * X - hi - lo) / span, -1.0, 1.0)
    sin_phi = np.sqrt(np.maximum(1.0 - z ** 2, 0.0))
    gasf = z[:, :, None] * z[:, None, :] - sin_phi[:, :, None] * sin_phi[:, None, :]
    gadf = sin_phi[:, :, None] * z[:, None, :] - z[:, :, None] * sin_phi[:, None, :]
    return np.stack([gasf, gadf], axis=-1).astype("float32")
