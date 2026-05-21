from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_predictor
from api.monitoring import (
    MODEL_INPUT_LAST_CLOSE,
    MODEL_PREDICTION_LATENCY,
    MODEL_PREDICTIONS,
    metrics_response,
)
from api.schemas import (
    ForecastResponse,
    HealthResponse,
    InfoResponse,
    PredictRequest,
    PredictResponse,
)
from src.data import fetch_recent_closes
from src.predict import Predictor

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_model=InfoResponse)
def info(predictor: Predictor = Depends(get_predictor)) -> InfoResponse:
    md = predictor.metadata
    return InfoResponse(
        ticker=md["ticker"],
        window=md["window"],
        feature=md["feature"],
        trained_at=md["trained_at"],
        metrics_test=md["metrics_test"],
        metrics_val=md["metrics_val"],
        framework=md["framework"],
        framework_version=md["framework_version"],
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/metrics", include_in_schema=False)
def metrics():
    return metrics_response()


@router.post("/predict", response_model=PredictResponse)
def predict(
    payload: PredictRequest,
    predictor: Predictor = Depends(get_predictor),
) -> PredictResponse:
    if len(payload.closes) < predictor.window:
        raise HTTPException(
            status_code=422,
            detail=f"need at least {predictor.window} closes, got {len(payload.closes)}",
        )

    MODEL_INPUT_LAST_CLOSE.set(payload.closes[-1])
    start = time.perf_counter()
    pred = predictor.predict_next(payload.closes)
    MODEL_PREDICTION_LATENCY.observe(time.perf_counter() - start)
    MODEL_PREDICTIONS.labels(endpoint="/predict").inc()

    return PredictResponse(
        ticker=predictor.metadata["ticker"],
        window=predictor.window,
        used_closes=predictor.window,
        prediction=pred,
    )


@router.get("/predict/next", response_model=PredictResponse)
def predict_next(
    predictor: Predictor = Depends(get_predictor),
) -> PredictResponse:
    ticker = predictor.metadata["ticker"]
    try:
        closes = fetch_recent_closes(ticker, predictor.window)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    MODEL_INPUT_LAST_CLOSE.set(closes[-1])
    start = time.perf_counter()
    pred = predictor.predict_next(closes)
    MODEL_PREDICTION_LATENCY.observe(time.perf_counter() - start)
    MODEL_PREDICTIONS.labels(endpoint="/predict/next").inc()

    return PredictResponse(
        ticker=ticker,
        window=predictor.window,
        used_closes=len(closes),
        prediction=pred,
    )


@router.get("/predict/forecast", response_model=ForecastResponse)
def predict_forecast(
    horizon: int = Query(5, ge=1, le=30, description="Número de dias futuros a prever"),
    predictor: Predictor = Depends(get_predictor),
) -> ForecastResponse:
    ticker = predictor.metadata["ticker"]
    try:
        closes = fetch_recent_closes(ticker, predictor.window)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    MODEL_INPUT_LAST_CLOSE.set(closes[-1])
    start = time.perf_counter()
    preds = predictor.forecast(closes, horizon=horizon)
    MODEL_PREDICTION_LATENCY.observe(time.perf_counter() - start)
    MODEL_PREDICTIONS.labels(endpoint="/predict/forecast").inc(horizon)

    return ForecastResponse(
        ticker=ticker,
        window=predictor.window,
        horizon=horizon,
        predictions=preds,
    )
