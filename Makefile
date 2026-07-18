.PHONY: install dev lint test train serve data clean docker-up docker-down mlflow

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
	pytest tests/ -x --cov=src --cov-report=term-missing --cov-fail-under=60

test-ci:
	pytest tests/ -x --cov=src --cov-report=xml --cov-fail-under=60 --junitxml=test-results.xml

train:
	python -m src.models.train

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

data:
	dvc pull
	python -m src.features.feature_engineering

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ test-results.xml coverage.xml

evaluate:
	python -m evaluation.ragas_eval
	python -m evaluation.llm_judge
