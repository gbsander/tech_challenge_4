.PHONY: install install-dev train test api docker-build docker-run lint clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

train:
	python -m src.train_pipeline

test:
	pytest -q

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t tech-challenge-4 .

docker-run:
	docker run --rm -p 8000:8000 --name tc4 tech-challenge-4

clean:
	rm -rf artifacts/model.keras artifacts/scaler.pkl artifacts/metadata.json
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
