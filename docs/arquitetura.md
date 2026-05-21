# Arquitetura

```
┌──────────────────────────────────────────────────────────────┐
│                    OFFLINE — pipeline de treino              │
│                                                              │
│   yfinance        ┌───────────────┐                          │
│   (PETR4.SA) ────►│  src/data.py  │                          │
│                   │  fetch +      │                          │
│                   │  split        │                          │
│                   │  temporal     │                          │
│                   │  +            │                          │
│                   │  janelamento  │                          │
│                   │  + scaler     │                          │
│                   │  (só treino)  │                          │
│                   └──────┬────────┘                          │
│                          │                                   │
│                   ┌──────▼─────────┐                         │
│                   │  src/model.py  │                         │
│                   │  Sequential:   │                         │
│                   │  LSTM(64)→     │                         │
│                   │  Dropout→      │                         │
│                   │  LSTM(32)→     │                         │
│                   │  Dropout→      │                         │
│                   │  Dense(16)→    │                         │
│                   │  Dense(1)      │                         │
│                   └──────┬─────────┘                         │
│                          │ EarlyStopping + ReduceLROnPlateau │
│                   ┌──────▼─────────┐                         │
│                   │ src/evaluate.py│                         │
│                   │ MAE/RMSE/MAPE  │                         │
│                   └──────┬─────────┘                         │
│                          │                                   │
│                   ┌──────▼────────────┐                      │
│                   │   artifacts/      │                      │
│                   │   - model.keras   │                      │
│                   │   - scaler.pkl    │                      │
│                   │   - metadata.json │                      │
│                   └───────────────────┘                      │
└────────────────────────────┬─────────────────────────────────┘
                             │ (commitados no git)
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                     ONLINE — API em produção                 │
│                                                              │
│   ┌─────────────┐                                            │
│   │  client     │──── HTTP ────►┌────────────────────┐       │
│   └─────────────┘               │   FastAPI app      │       │
│                                 │   (uvicorn)        │       │
│                                 │  ┌──────────────┐  │       │
│                                 │  │ Prometheus   │  │       │
│                                 │  │ Middleware   │  │       │
│                                 │  └──────┬───────┘  │       │
│                                 │  ┌──────▼───────┐  │       │
│                                 │  │   routes.py  │  │       │
│                                 │  └──────┬───────┘  │       │
│                                 │  ┌──────▼───────┐  │       │
│                                 │  │ Predictor    │  │       │
│                                 │  │ (singleton,  │  │       │
│                                 │  │  app.state)  │  │       │
│                                 │  └──────────────┘  │       │
│                                 │                    │       │
│                                 │  /metrics ─────────┼───►   │
│                                 │  Prometheus exposition     │
│                                 └────────────────────┘       │
│                                                              │
│   Container: Docker (Python 3.11-slim, multi-stage)          │
│   Deploy:    Render.com (auto-deploy via GitHub)             │
└──────────────────────────────────────────────────────────────┘
```

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | Metadata do modelo (ticker, métricas, treino) |
| `GET` | `/health` | Liveness probe (Render usa pra healthcheck) |
| `GET` | `/metrics` | Prometheus exposition format |
| `GET` | `/docs` | Swagger UI gerado pelo FastAPI |
| `POST` | `/predict` | Recebe N closes e retorna próxima predição |
| `GET` | `/predict/next` | Fetch automático yfinance → predição |
| `GET` | `/predict/forecast?horizon=N` | Forecast multi-step autoregressivo |

## Métricas Prometheus expostas

- `http_requests_total{method,route,status}` — counter
- `http_request_duration_seconds{route}` — histogram
- `model_predictions_total{endpoint}` — counter
- `model_prediction_latency_seconds` — histogram (só inferência)
- `model_input_last_close` — gauge (sanity check de drift do input)
- `process_*` — métricas de runtime do `prometheus_client`

## Decisões de projeto

| Decisão | Por quê |
|---|---|
| Univariate (só `Close`) | Escopo mais enxuto, suficiente pra demonstrar o pipeline LSTM |
| `MinMaxScaler` fittado **só no treino** | Evitar data leakage |
| Split **temporal** (não aleatório) | Séries temporais não podem ser embaralhadas — futuro não pode treinar o passado |
| Janela de 60 dias | Padrão da literatura LSTM-stocks; ~3 meses de pregão |
| EarlyStopping(patience=8) | Evita overfit; backup com ReduceLROnPlateau quando estagna |
| `tensorflow` (Mac) / `tensorflow-cpu` (Linux) | Wheel `-cpu` reduz ~600 MB da imagem; só existe em Linux |
| Render.com | Free tier suporta Docker, auto-deploy via blueprint |
