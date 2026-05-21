from __future__ import annotations

import time

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests by route, method and status",
    labelnames=("method", "route", "status"),
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    labelnames=("route",),
)
MODEL_PREDICTIONS = Counter(
    "model_predictions_total",
    "Total predictions served by the model",
    labelnames=("endpoint",),
)
MODEL_PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Time spent inside model.predict() per call",
)
MODEL_INPUT_LAST_CLOSE = Gauge(
    "model_input_last_close",
    "Last close value received as input — sanity check for input drift",
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        route_path = request.url.path
        start = time.perf_counter()
        try:
            response = await call_next(request)
            status = str(response.status_code)
        except Exception:
            HTTP_REQUESTS.labels(method=request.method, route=route_path, status="500").inc()
            raise
        finally:
            elapsed = time.perf_counter() - start
            HTTP_LATENCY.labels(route=route_path).observe(elapsed)
        HTTP_REQUESTS.labels(method=request.method, route=route_path, status=status).inc()
        return response


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
