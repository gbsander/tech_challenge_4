from __future__ import annotations

from fastapi import Request

from src.predict import Predictor


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor
