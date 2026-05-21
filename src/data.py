from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler
    raw_close: pd.Series


def fetch_history(ticker: str, start_date: str, end_date: str | None = None) -> pd.Series:
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned empty dataframe for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"].dropna()
    close.name = "close"
    return close


def fetch_recent_closes(
    ticker: str,
    n_closes: int,
    lookback_days: int | None = None,
    retries: int = 4,
    pause: float = 1.5,
) -> list[float]:
    """Busca os últimos ``n_closes`` fechamentos do ticker via yfinance.

    O Yahoo Finance bloqueia/limita IPs de datacenter intermitentemente
    (comum no Render). Por isso tentamos ``retries`` vezes com backoff antes
    de desistir, em vez de falhar na primeira resposta vazia.

    Levanta ``RuntimeError`` se não conseguir dados suficientes após as
    tentativas — o chamador converte em HTTP 502.
    """
    lookback_days = lookback_days or n_closes * 3
    last_err: str = "no data"
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker,
                period=f"{lookback_days}d",
                auto_adjust=False,
                progress=False,
            )
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                closes = df["Close"].dropna().tail(n_closes).tolist()
                if len(closes) >= n_closes:
                    return [float(c) for c in closes]
                last_err = f"only got {len(closes)} closes, need {n_closes}"
            else:
                last_err = "empty dataframe"
        except Exception as exc:  # noqa: BLE001 - rede/parse flutuam
            last_err = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(pause * attempt)
    raise RuntimeError(f"yfinance failed for {ticker} after {retries} tries: {last_err}")


def temporal_split(
    series: pd.Series, train_frac: float, val_frac: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    n = len(series)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return series.iloc[:train_end], series.iloc[train_end:val_end], series.iloc[val_end:]


def make_windows(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if len(values) <= window:
        raise ValueError(f"need at least {window + 1} points, got {len(values)}")
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window : i])
        y.append(values[i])
    X = np.asarray(X, dtype=np.float32).reshape(-1, window, 1)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return X, y


def build_dataset(
    ticker: str,
    start_date: str,
    end_date: str | None,
    window: int,
    train_frac: float,
    val_frac: float,
) -> Dataset:
    close = fetch_history(ticker, start_date, end_date)
    train_s, val_s, test_s = temporal_split(close, train_frac, val_frac)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_s.values.reshape(-1, 1)).ravel()
    val_scaled = scaler.transform(val_s.values.reshape(-1, 1)).ravel()
    test_scaled = scaler.transform(test_s.values.reshape(-1, 1)).ravel()

    train_seq = train_scaled
    val_seq = np.concatenate([train_scaled[-window:], val_scaled])
    test_seq = np.concatenate([val_scaled[-window:], test_scaled])

    X_train, y_train = make_windows(train_seq, window)
    X_val, y_val = make_windows(val_seq, window)
    X_test, y_test = make_windows(test_seq, window)

    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        raw_close=close,
    )
