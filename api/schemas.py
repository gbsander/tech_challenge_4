from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    closes: Annotated[list[float], Field(min_length=1, description="Sequência de preços de fechamento; usa-se os últimos N=window")]

    @field_validator("closes")
    @classmethod
    def _positive(cls, v: list[float]) -> list[float]:
        if any(x <= 0 for x in v):
            raise ValueError("all closes must be > 0")
        return v


class PredictResponse(BaseModel):
    ticker: str
    window: int
    used_closes: int
    prediction: float


class ForecastResponse(BaseModel):
    ticker: str
    window: int
    horizon: int
    predictions: list[float]


class HealthResponse(BaseModel):
    status: str = "ok"


class InfoResponse(BaseModel):
    ticker: str
    window: int
    feature: str
    trained_at: str
    metrics_test: dict[str, float]
    metrics_val: dict[str, float]
    framework: str
    framework_version: str
