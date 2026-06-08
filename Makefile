.PHONY: setup install lint format test preprocess train-xgb train-graphsage train-gat demo api dashboard mlflow docker-build docker-up docker-down security-scan clean help

RED    := \033[0;31m
GREEN  := \033[0;32m
YELLOW := \033[1;33m
NC     := \033[0m

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

setup: ## Initial project setup (uv sync + pre-commit)
	uv sync
	pre-commit install
	dvc init --no-scm || true
	@echo "$(GREEN)✓ Setup complete$(NC)"

install: ## Install production dependencies only
	uv sync --no-dev

lint: ## Run all linters (ruff + mypy + bandit)
	ruff check src tests
	ruff format --check src tests
	mypy src
	bandit -r src -c pyproject.toml

format: ## Auto-format code
	ruff format src tests
	ruff check --fix src tests

test: ## Run tests with coverage
	pytest tests/ -v --cov=src/c2gnn --cov-report=term-missing

preprocess: ## Run data preprocessing pipeline
	python -m c2gnn.data.preprocess \
		--input data/raw/ctu13 \
		--output data/processed \
		--scenario 10

train-xgb: ## Train XGBoost baseline
	python -m c2gnn.models.train \
		--model xgboost \
		--config configs/model_xgboost.yaml

train-graphsage: ## Train GraphSAGE model
	python -m c2gnn.models.train \
		--model graphsage \
		--config configs/model_graphsage.yaml

train-gat: ## Train GATv2 model
	python -m c2gnn.models.train \
		--model gat \
		--config configs/model_gat.yaml

demo: ## Run realtime simulation demo
	python -m c2gnn.realtime.pipeline \
		--data data/processed/ctu13_scenario10_flows.parquet \
		--model models/artifacts/graphsage_best.pt \
		--realtime-factor 20.0

api: ## Start FastAPI server
	uvicorn c2gnn.api:app --host 0.0.0.0 --port 8000 --reload

dashboard: ## Start Streamlit dashboard
	streamlit run src/c2gnn/dashboard/app.py

mlflow: ## Start MLflow UI
	mlflow ui --port 5000

docker-build: ## Build Docker image
	docker build -t c2gnn:latest .

docker-up: ## Start all services via docker-compose
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

security-scan: ## Run security scans (bandit + trivy)
	bandit -r src -f json -o reports/bandit_report.json || true
	trivy fs . --exit-code 0 --format table

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov coverage.xml
