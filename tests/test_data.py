import numpy as np
import pandas as pd
import pytest

import src.data as data_mod
from src.data import fetch_recent_closes, make_windows, temporal_split


def test_temporal_split_proportions_and_order():
    s = pd.Series(np.arange(100, dtype=float))
    train, val, test = temporal_split(s, train_frac=0.8, val_frac=0.1)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10
    # No leakage / ordering preserved
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()


def test_make_windows_shape_and_alignment():
    values = np.arange(10, dtype=np.float32)
    X, y = make_windows(values, window=3)
    assert X.shape == (7, 3, 1)
    assert y.shape == (7, 1)
    # window starting at index 0..2 predicts index 3
    np.testing.assert_array_equal(X[0].ravel(), [0, 1, 2])
    assert y[0, 0] == 3.0


def test_make_windows_too_short():
    with pytest.raises(ValueError):
        make_windows(np.arange(3, dtype=np.float32), window=3)


def _fake_df(n):
    return pd.DataFrame({"Close": np.linspace(30, 40, n)})


def test_fetch_recent_closes_retries_then_succeeds(monkeypatch):
    """Primeira chamada vazia, segunda OK — retry deve recuperar."""
    calls = {"n": 0}

    def fake_download(*args, **kwargs):
        calls["n"] += 1
        return pd.DataFrame() if calls["n"] == 1 else _fake_df(100)

    monkeypatch.setattr(data_mod.yf, "download", fake_download)
    monkeypatch.setattr(data_mod.time, "sleep", lambda *_: None)

    closes = fetch_recent_closes("PETR4.SA", n_closes=60, retries=3)
    assert len(closes) == 60
    assert calls["n"] == 2
    assert all(isinstance(c, float) for c in closes)


def test_fetch_recent_closes_raises_after_retries(monkeypatch):
    monkeypatch.setattr(data_mod.yf, "download", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(data_mod.time, "sleep", lambda *_: None)

    with pytest.raises(RuntimeError, match="after 3 tries"):
        fetch_recent_closes("PETR4.SA", n_closes=60, retries=3)


def test_fetch_recent_closes_survives_exception(monkeypatch):
    """Se yf.download lançar exceção, retry continua em vez de propagar."""
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("Expecting value: line 1 column 1")
        return _fake_df(100)

    monkeypatch.setattr(data_mod.yf, "download", flaky)
    monkeypatch.setattr(data_mod.time, "sleep", lambda *_: None)

    closes = fetch_recent_closes("PETR4.SA", n_closes=60, retries=3)
    assert len(closes) == 60
