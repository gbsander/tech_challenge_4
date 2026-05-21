import numpy as np

from src.evaluate import compute_metrics
from src.model import build_model


def test_build_model_io_shapes():
    m = build_model(window=10, n_features=1)
    assert m.input_shape == (None, 10, 1)
    assert m.output_shape == (None, 1)


def test_model_trains_one_epoch_on_fake_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 10, 1)).astype(np.float32)
    y = rng.normal(size=(64, 1)).astype(np.float32)
    m = build_model(window=10, n_features=1)
    m.fit(X, y, epochs=1, batch_size=16, verbose=0)


def test_compute_metrics_basic():
    y_true = np.array([100.0, 110.0, 105.0])
    y_pred = np.array([102.0, 108.0, 107.0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["mae"] == 2.0
    assert metrics["rmse"] > 0
    assert metrics["mape"] > 0
