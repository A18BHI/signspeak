"""Store labeled samples, train a classifier, and predict live signs."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_SIZE

DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLES_PATH = DATA_DIR / "samples.npz"
MODEL_PATH = DATA_DIR / "model.joblib"
META_PATH = DATA_DIR / "meta.json"


class SignTrainer:
    def __init__(self) -> None:
        self.X: list[np.ndarray] = []
        self.y: list[str] = []
        self.model: Pipeline | None = None
        self._history: deque[str] = deque(maxlen=8)
        self.load()

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for label in self.y:
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))

    def labels(self) -> list[str]:
        return list(self.counts().keys())

    def add_samples(self, label: str, samples: list[np.ndarray]) -> int:
        name = label.strip().lower()
        if not name or not samples:
            return 0
        for sample in samples:
            if sample is None or sample.shape != (FEATURE_SIZE,):
                continue
            self.X.append(sample.astype(np.float32))
            self.y.append(name)
        self.save_samples()
        return len(samples)

    def delete_label(self, label: str) -> None:
        name = label.strip().lower()
        kept_x, kept_y = [], []
        for x, y in zip(self.X, self.y):
            if y != name:
                kept_x.append(x)
                kept_y.append(y)
        self.X, self.y = kept_x, kept_y
        self.model = None
        self.save_samples()
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()

    def can_train(self) -> bool:
        return len(set(self.y)) >= 2 and len(self.y) >= 20

    def train(self) -> dict:
        if not self.can_train():
            raise ValueError("Need at least 2 signs and 20 samples to train.")

        X = np.stack(self.X)
        y = np.array(self.y)
        n_labels = len(set(self.y))

        forest = RandomForestClassifier(
            n_estimators=220,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", forest),
            ]
        )
        self.model.fit(X, y)
        acc = float(self.model.score(X, y))

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        META_PATH.write_text(
            json.dumps({"train_accuracy": acc, "n_samples": int(len(y)), "labels": self.labels()}, indent=2),
            encoding="utf-8",
        )
        return {"accuracy": acc, "samples": len(y), "signs": n_labels}

    def predict(self, features: np.ndarray | None) -> tuple[str | None, float]:
        if self.model is None or features is None:
            self._history.clear()
            return None, 0.0

        proba = self.model.predict_proba([features])[0]
        idx = int(np.argmax(proba))
        label = str(self.model.classes_[idx])
        confidence = float(proba[idx])

        self._history.append(label)
        if self._history.count(label) < max(3, len(self._history) // 2):
            return None, confidence
        return label, confidence

    def save_samples(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not self.X:
            if SAMPLES_PATH.exists():
                SAMPLES_PATH.unlink()
            return
        np.savez_compressed(SAMPLES_PATH, X=np.stack(self.X), y=np.array(self.y))

    def load(self) -> None:
        if SAMPLES_PATH.exists():
            blob = np.load(SAMPLES_PATH, allow_pickle=True)
            self.X = [row.astype(np.float32) for row in blob["X"]]
            self.y = [str(label) for label in blob["y"]]
        if MODEL_PATH.exists():
            try:
                self.model = joblib.load(MODEL_PATH)
            except Exception:
                self.model = None
