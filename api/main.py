from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pythonjsonlogger import jsonlogger

from api.monitoring import PrometheusMiddleware
from api.routes import router
from src.predict import Predictor


def _configure_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "info").upper()
    handler = logging.StreamHandler()
    handler.setFormatter(
        jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = logging.getLogger(__name__)
    logger.info("loading_predictor")
    app.state.predictor = Predictor()
    logger.info(
        "predictor_loaded",
        extra={"ticker": app.state.predictor.metadata.get("ticker"), "window": app.state.predictor.window},
    )
    yield


def create_app() -> FastAPI:
    _configure_logging()
    app = FastAPI(
        title="LSTM Stock Predictor — PETR4",
        description="API REST que serve um modelo LSTM treinado para prever o fechamento de PETR4.SA.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(PrometheusMiddleware)
    app.include_router(router)
    return app


app = create_app()
