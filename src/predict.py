from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from tensorflow import keras

from src.config import settings


class Predictor:
    """Carrega artefatos uma vez e expõe inferência single-step e multi-step."""

    def __init__(
        self,
        model_path: Path | None = None,
        scaler_path: Path | None = None,
        metadata_path: Path | None = None,
    ) -> None:
        self.model_path = model_path or settings.model_path
        self.scaler_path = scaler_path or settings.scaler_path
        self.metadata_path = metadata_path or settings.metadata_path

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"model not found at {self.model_path} — run `python -m src.train_pipeline` first"
            )

        self.model = keras.models.load_model(self.model_path)
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata: dict = json.load(f)

        self.window: int = int(self.metadata["window"])

    def _to_scaled_window(self, closes: list[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(closes, dtype=np.float32).ravel()
        if arr.size < self.window:
            raise ValueError(f"need at least {self.window} closes, got {arr.size}")
        last = arr[-self.window :].reshape(-1, 1)
        scaled = self.scaler.transform(last).reshape(1, self.window, 1).astype(np.float32)
        return scaled

    def predict_next(self, closes: list[float] | np.ndarray) -> float:
        x = self._to_scaled_window(closes)
        scaled_pred = self.model.predict(x, verbose=0)
        return float(self.scaler.inverse_transform(scaled_pred).ravel()[0])

    def forecast(self, closes: list[float] | np.ndarray, horizon: int) -> list[float]:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        arr = np.asarray(closes, dtype=np.float32).ravel()
        if arr.size < self.window:
            raise ValueError(f"need at least {self.window} closes, got {arr.size}")

        scaled = self.scaler.transform(arr.reshape(-1, 1)).ravel()
        window = scaled[-self.window :].copy()
        preds_scaled: list[float] = []
        for _ in range(horizon):
            x = window.reshape(1, self.window, 1).astype(np.float32)
            yhat = float(self.model.predict(x, verbose=0).ravel()[0])
            preds_scaled.append(yhat)
            window = np.concatenate([window[1:], [yhat]])

        preds = self.scaler.inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).ravel()
        return [float(v) for v in preds]
