"""Evaluate the saved landmark MLP exactly once on the untouched test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from train import load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    classes = json.loads((args.models_dir / "classes.json").read_text(encoding="utf-8"))
    x_test, y_test = load_split(args.processed_dir / "test.csv", classes)
    scaler = joblib.load(args.models_dir / "scaler.pkl")
    model = tf.keras.models.load_model(args.models_dir / "best_model.keras")
    loss, accuracy = model.evaluate(scaler.transform(x_test), y_test, verbose=0)
    predictions = model.predict(scaler.transform(x_test), verbose=0).argmax(axis=1)
    report = classification_report(y_test, predictions, labels=np.arange(len(classes)), target_names=classes, digits=4, zero_division=0)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="weighted", zero_division=0)
    metrics = {"test_loss": float(loss), "test_accuracy": float(accuracy), "weighted_precision": float(precision), "weighted_recall": float(recall), "weighted_f1": float(f1)}
    (args.results_dir / "classification_report.txt").write_text(report + "\n" + json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.results_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    matrix = confusion_matrix(y_test, predictions, labels=np.arange(len(classes)))
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=axis)
    axis.set(xlabel="Predicted label", ylabel="True label", title="Test-set confusion matrix")
    figure.tight_layout()
    figure.savefig(args.results_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
