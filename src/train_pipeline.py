from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone

import numpy as np

from src.config import ARTIFACTS_DIR, settings
from src.data import build_dataset
from src.evaluate import compute_metrics
from src.model import build_model, train


def run(
    ticker: str,
    start_date: str,
    end_date: str | None,
    window: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    train_frac: float,
    val_frac: float,
) -> dict:
    print(f"[1/5] Fetching {ticker} from {start_date} to {end_date or 'today'}")
    ds = build_dataset(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        window=window,
        train_frac=train_frac,
        val_frac=val_frac,
    )
    print(
        f"  train={ds.X_train.shape}  val={ds.X_val.shape}  test={ds.X_test.shape}"
        f"  raw_close={len(ds.raw_close)} pts"
    )

    print(f"[2/5] Building LSTM (window={window})")
    model = build_model(window=window, n_features=1, learning_rate=learning_rate)
    model.summary()

    print(f"[3/5] Training (epochs={epochs}, batch={batch_size})")
    history = train(
        model=model,
        X_train=ds.X_train,
        y_train=ds.y_train,
        X_val=ds.X_val,
        y_val=ds.y_val,
        epochs=epochs,
        batch_size=batch_size,
    )

    print("[4/5] Evaluating on test")
    test_pred_scaled = model.predict(ds.X_test, verbose=0)
    test_pred = ds.scaler.inverse_transform(test_pred_scaled).ravel()
    test_true = ds.scaler.inverse_transform(ds.y_test).ravel()
    metrics = compute_metrics(test_true, test_pred)
    print(f"  test  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  MAPE={metrics['mape']:.2f}%")

    val_pred_scaled = model.predict(ds.X_val, verbose=0)
    val_pred = ds.scaler.inverse_transform(val_pred_scaled).ravel()
    val_true = ds.scaler.inverse_transform(ds.y_val).ravel()
    val_metrics = compute_metrics(val_true, val_pred)
    print(f"  val   MAE={val_metrics['mae']:.4f}  RMSE={val_metrics['rmse']:.4f}  MAPE={val_metrics['mape']:.2f}%")

    print("[5/5] Saving artifacts")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(settings.model_path)
    with open(settings.scaler_path, "wb") as f:
        pickle.dump(ds.scaler, f)

    metadata = {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "window": window,
        "feature": "Close",
        "n_train_samples": int(ds.X_train.shape[0]),
        "n_val_samples": int(ds.X_val.shape[0]),
        "n_test_samples": int(ds.X_test.shape[0]),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "epochs_run": int(len(history.history["loss"])),
        "metrics_test": metrics,
        "metrics_val": val_metrics,
        "framework": "tensorflow",
        "framework_version": __import__("tensorflow").__version__,
    }
    with open(settings.metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  -> {settings.model_path}")
    print(f"  -> {settings.scaler_path}")
    print(f"  -> {settings.metadata_path}")
    return metadata


def main() -> None:
    p = argparse.ArgumentParser(description="Train LSTM stock predictor")
    p.add_argument("--ticker", default=settings.ticker)
    p.add_argument("--start", default=settings.start_date)
    p.add_argument("--end", default=None)
    p.add_argument("--window", type=int, default=settings.window)
    p.add_argument("--epochs", type=int, default=settings.epochs)
    p.add_argument("--batch", type=int, default=settings.batch_size)
    p.add_argument("--lr", type=float, default=settings.learning_rate)
    p.add_argument("--train-frac", type=float, default=settings.train_frac)
    p.add_argument("--val-frac", type=float, default=settings.val_frac)
    args = p.parse_args()

    run(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        window=args.window,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )


if __name__ == "__main__":
    main()
