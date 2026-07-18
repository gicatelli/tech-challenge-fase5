.PHONY: install dev lint test train serve data clean docker-up docker-down mlflow evaluate ingest drift

# ============================================
# Datathon Fase 05 - Makefile
# ============================================

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src/ tests/ evaluation/
	mypy src/ --ignore-missing-imports

test:
	pytest tests/ -x --cov=src --cov-report=term-missing --cov-fail-under=30

test-ci:
	pytest tests/ -x --cov=src --cov-report=xml --cov-fail-under=30 --junitxml=test-results.xml

train:
	python -m src.models.train

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

data:
	python src/data_collection.py

ingest:
	python -m src.agent.rag_pipeline ingest

drift:
	python -m src.monitoring.drift

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

evaluate:
	python -m evaluation.ragas_eval
	python -m evaluation.llm_judge
	python -m evaluation.ab_test_prompts

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ test-results.xml coverage.xml
