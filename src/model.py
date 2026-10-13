"""Train a surface classifier and score precision, recall, and latency."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from dataset import features


def matrix(images: list[np.ndarray]) -> np.ndarray:
    return np.vstack([features(image) for image in images])


def train(images: list[np.ndarray], labels: np.ndarray) -> RandomForestClassifier:
    forest = RandomForestClassifier(
        n_estimators=80,
        max_depth=8,
        random_state=7,
        n_jobs=1,
    )
    forest.fit(matrix(images), labels)
    return forest


def predict_many(model: RandomForestClassifier, images: list[np.ndarray]) -> np.ndarray:
    return model.predict(matrix(images))


def metrics(labels: np.ndarray, predicted: np.ndarray, elapsed_s: float) -> dict:
    labels = labels.astype(bool)
    predicted = predicted.astype(bool)
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    tn = int(np.sum(~predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    false_alarm = fp / max(fp + tn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "false_alarm_rate": false_alarm,
        "accuracy": (tp + tn) / max(len(labels), 1),
        "latency_ms": 1000 * elapsed_s / max(len(labels), 1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def timed_predict(model: RandomForestClassifier, images: list[np.ndarray]) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    predicted = predict_many(model, images)
    return predicted, time.perf_counter() - start


def save_model(model: RandomForestClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> RandomForestClassifier:
    return joblib.load(path)
