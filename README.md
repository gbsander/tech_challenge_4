# LSTM Stock Predictor — PETR4

Previsão do preço de fechamento da PETR4 (Petrobras, B3) com uma rede LSTM,
servida por uma API REST. Projeto da Fase 4 da pós em Machine Learning
Engineering.

API em produção: https://tech-challenge-4-api-o61w.onrender.com

## Como rodar

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# treina e gera os artefatos em artifacts/
python -m src.train_pipeline

# sobe a API (Swagger em http://localhost:8000/docs)
uvicorn api.main:app --reload

# testes
pytest -q
```

Os atalhos do `Makefile` (`make train`, `make api`, `make test`,
`make docker-build`) fazem o mesmo. O treino leva cerca de 50s em CPU.

## Estrutura

```
src/                  pipeline de treino
  config.py           parâmetros (ticker, janela, learning rate...)
  data.py             coleta via yfinance, split temporal, janelamento, scaler
  model.py            arquitetura LSTM e loop de treino
  evaluate.py         MAE, RMSE, MAPE
  predict.py          carrega o modelo salvo e faz inferência
  train_pipeline.py   roda tudo e salva model.keras + scaler.pkl + metadata.json
api/                  FastAPI
  main.py             criação do app e carga do modelo no startup
  routes.py           endpoints
  schemas.py          modelos Pydantic de request/response
  monitoring.py       middleware de métricas Prometheus
artifacts/            modelo treinado (versionado no repo)
notebooks/            exploração e demo da API
tests/                pytest
docs/arquitetura.md   diagrama do pipeline
Dockerfile            imagem multi-stage
render.yaml           configuração do deploy no Render
```

## Modelagem

- Histórico de 2015 até hoje, só a coluna `Close`.
- Split temporal 80/10/10, sem embaralhar — treino no passado, teste no futuro.
- `MinMaxScaler` ajustado apenas no treino, para não vazar informação.
- Janela de 60 pregões prevendo o fechamento do dia seguinte.
- Arquitetura: `LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(16) → Dense(1)`.
- Otimizador Adam, perda MSE, EarlyStopping e ReduceLROnPlateau.

Resultado do último treino (em `artifacts/metadata.json`):

| Conjunto  | MAE (R$) | RMSE (R$) | MAPE  |
|-----------|---------:|----------:|------:|
| Validação |     0.96 |      1.19 | 2.54% |
| Teste     |     0.97 |      1.31 | 2.69% |

As métricas de validação e teste próximas indicam que não houve overfit. Vale
lembrar que é previsão de um dia à frente, onde o preço de amanhã tende a ficar
perto do de hoje — o objetivo do trabalho é o pipeline completo, não bater o
mercado.

## API

Swagger em `/docs`, ReDoc em `/redoc`.

| Método | Rota                  | Descrição                                  |
|--------|-----------------------|--------------------------------------------|
| GET    | `/`                   | metadados do modelo                        |
| GET    | `/health`             | liveness                                   |
| GET    | `/metrics`            | métricas no formato Prometheus             |
| POST   | `/predict`            | recebe `{"closes": [...]}` (≥ 60) e prevê  |
| GET    | `/predict/next`       | busca os últimos 60 fechamentos e prevê    |
| GET    | `/predict/forecast`   | `?horizon=N` (1–30), previsão multi-step   |

```bash
curl http://localhost:8000/health
curl http://localhost:8000/predict/next
curl 'http://localhost:8000/predict/forecast?horizon=5'

python -c "import json; print(json.dumps({'closes':[37.5]*60}))" \
  | curl -s -X POST http://localhost:8000/predict \
      -H 'content-type: application/json' -d @-
```

O Pydantic devolve 422 quando há menos de 60 fechamentos ou algum valor não
positivo.

## Monitoramento

`api/monitoring.py` adiciona um middleware que expõe em `/metrics`:

- `http_requests_total` — requests por rota, método e status
- `http_request_duration_seconds` — latência das requisições
- `model_predictions_total` — previsões servidas
- `model_prediction_latency_seconds` — tempo dentro do `model.predict()`
- `model_input_last_close` — último valor recebido, para checar drift de entrada

Para coletar com Prometheus:

```yaml
scrape_configs:
  - job_name: 'lstm-api'
    metrics_path: /metrics
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
```

Os logs saem em JSON, prontos para um agregador.

## Deploy

O `render.yaml` descreve o serviço (Docker, healthcheck em `/health`,
auto-deploy a cada push). No painel do Render: New → Blueprint → escolher o
repositório. O primeiro build leva alguns minutos por causa do TensorFlow. No
plano gratuito o serviço hiberna após 15 min ociosos, então a primeira
requisição depois disso pode demorar até um minuto.

Build local antes de subir:

```bash
make docker-build
make docker-run
```

## Possíveis evoluções

- Usar mais features (OHLCV, indicadores técnicos) além do fechamento.
- Retreino agendado por GitHub Actions.
- Detecção de drift comparando a distribuição das entradas com a do treino.
- Comparar com outras arquiteturas (GRU, modelos clássicos de série temporal).

## Stack

TensorFlow/Keras, FastAPI, Uvicorn, yfinance, pandas, scikit-learn,
prometheus-client. Container em Python 3.11-slim, deploy no Render.

Dados via Yahoo Finance. Uso educacional.
