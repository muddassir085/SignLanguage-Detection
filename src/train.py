"""Train the landmark MLP using training-only feature scaling."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from landmarks import FEATURE_SIZE, feature_names


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_split(path: Path, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    features = frame.loc[:, feature_names()].to_numpy(dtype=np.float32)
    if features.shape[1] != FEATURE_SIZE:
        raise ValueError(f"Expected {FEATURE_SIZE} features, got {features.shape[1]}")
    labels = frame["label"].map({name: index for index, name in enumerate(classes)})
    if labels.isna().any():
        raise ValueError(f"Unknown label found in {path}")
    return features, labels.to_numpy(dtype=np.int64)


def build_model(input_size: int, class_count: int, learning_rate: float) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_size,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(class_count, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def save_history_plot(history: tf.keras.callbacks.History, destination: Path) -> None:
    values, epochs = history.history, range(1, len(history.history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric, axis, title in (("accuracy", axes[0], "Accuracy"), ("loss", axes[1], "Loss")):
        axis.plot(epochs, values[metric], label="train")
        axis.plot(epochs, values[f"val_{metric}"], label="validation")
        axis.set(title=title, xlabel="Epoch", ylabel=title)
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    classes = json.loads((args.processed_dir / "classes.json").read_text(encoding="utf-8"))
    x_train, y_train = load_split(args.processed_dir / "train.csv", classes)
    x_validation, y_validation = load_split(args.processed_dir / "validation.csv", classes)
    scaler = StandardScaler().fit(x_train)
    joblib.dump(scaler, args.models_dir / "scaler.pkl")
    (args.models_dir / "classes.json").write_text(json.dumps(classes, indent=2) + "\n", encoding="utf-8")
    model = build_model(x_train.shape[1], len(classes), args.learning_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(args.models_dir / "best_model.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=max(2, args.patience // 3), factor=0.5),
    ]
    history = model.fit(scaler.transform(x_train), y_train, validation_data=(scaler.transform(x_validation), y_validation), epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=2)
    (args.results_dir / "training_history.json").write_text(json.dumps(history.history, indent=2) + "\n", encoding="utf-8")
    save_history_plot(history, args.results_dir / "training_history.png")
    print(f"Saved best model to {args.models_dir / 'best_model.keras'}")


if __name__ == "__main__":
    main()
