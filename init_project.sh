#!/usr/bin/env bash
# =============================================================================
# init_project.sh — Khởi tạo project realtime-c2-gnn-detection
# Chạy lần đầu sau khi tạo GitHub repo rỗng
# Yêu cầu: git, uv (pip install uv), Python 3.11+
# =============================================================================
set -euo pipefail

PROJECT_NAME="realtime-c2-gnn-detection"
PYTHON_VERSION="3.11"

echo "╔══════════════════════════════════════════════════╗"
echo "║  Initializing: $PROJECT_NAME"
echo "╚══════════════════════════════════════════════════╝"

# ─── 0. Check prerequisites ───────────────────────────────────────────────────
command -v git  &>/dev/null || { echo "ERROR: git not found"; exit 1; }
command -v uv   &>/dev/null || { echo "Installing uv..."; pip install uv; }

# ─── 1. Directory structure ───────────────────────────────────────────────────
echo "→ Creating directory structure..."
mkdir -p \
  .github/workflows \
  configs \
  data/{raw,interim,processed} \
  notebooks \
  src/c2gnn/{data,graph,models,realtime,evaluation,utils} \
  tests \
  scripts \
  reports/{figures,tables} \
  slides \
  docs

# ─── 2. Python project via uv ─────────────────────────────────────────────────
echo "→ Initializing uv project..."
uv init --python $PYTHON_VERSION --no-workspace .

# ─── 3. pyproject.toml ────────────────────────────────────────────────────────
echo "→ Writing pyproject.toml..."
cat > pyproject.toml << 'PYPROJECT'
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "c2gnn"
version = "0.1.0"
description = "Real-time C2 Traffic Detection using Graph Neural Networks"
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.11"
authors = [{ name = "Your Team", email = "team@example.com" }]

dependencies = [
    # Core ML
    "torch>=2.3.0",
    "xgboost>=2.1.0",
    "scikit-learn>=1.5.0",
    "shap>=0.45.0",

    # Data
    "pandas>=2.2.0",
    "polars>=0.20.0",
    "numpy>=1.26.0",
    "pyarrow>=16.0.0",

    # Graph
    "networkx>=3.3",
    "pyvis>=0.3.2",

    # Config & CLI
    "omegaconf>=2.3.0",
    "typer[all]>=0.12.0",
    "rich>=13.7.0",

    # API & Dashboard
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.30.0",
    "pydantic>=2.7.0",
    "streamlit>=1.35.0",
    "plotly>=5.22.0",

    # MLOps
    "mlflow>=2.13.0",
    "dvc>=3.51.0",

    # Logging
    "structlog>=24.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2.0",
    "pytest-cov>=5.0.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.4.7",
    "mypy>=1.10.0",
    "pre-commit>=3.7.0",
    "bandit[toml]>=1.7.0",
    "hypothesis>=6.103.0",
    "ipykernel>=6.29.0",
    "jupyterlab>=4.2.0",
]

[project.scripts]
c2gnn = "c2gnn.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/c2gnn"]

# ─── Ruff (replaces black + isort + flake8) ───────────────────────────────────
[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src"]

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM"]
ignore = ["E501"]

[tool.ruff.lint.isort]
known-first-party = ["c2gnn"]

# ─── Mypy ──────────────────────────────────────────────────────────────────────
[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true
warn_return_any = false

# ─── Pytest ────────────────────────────────────────────────────────────────────
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=src/c2gnn --cov-report=term-missing --cov-report=xml -v"

# ─── Bandit ────────────────────────────────────────────────────────────────────
[tool.bandit]
targets = ["src"]
skips = ["B101"]  # allow assert in tests
PYPROJECT

# ─── 4. Install PyTorch Geometric separately (needs extra index) ───────────────
echo "→ Installing dependencies..."
uv add torch==2.3.0
uv add torch-geometric \
    --extra-index-url https://data.pyg.org/whl/torch-2.3.0+cpu.html
uv add --dev pytest pytest-cov ruff mypy pre-commit bandit
uv sync

# ─── 5. .gitignore ────────────────────────────────────────────────────────────
echo "→ Writing .gitignore..."
cat > .gitignore << 'GITIGNORE'
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/
.eggs/
*.egg
*.whl

# Environments
.venv/
venv/
env/
.env
.env.*
!.env.example

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Data (tracked by DVC)
data/raw/
data/interim/
data/processed/
*.pcap
*.binetflow
*.parquet

# Models (tracked by DVC/MLflow)
models/artifacts/
*.pt
*.pkl
*.joblib
mlruns/
.mlflow/

# Reports (keep figures)
reports/tables/*.csv

# Notebooks checkpoints
.ipynb_checkpoints/
*.ipynb_checkpoints

# Testing
.coverage
coverage.xml
htmlcov/
.pytest_cache/
.ruff_cache/
.mypy_cache/

# Docker
.docker/

# OS
.DS_Store
Thumbs.db
GITIGNORE

# ─── 6. .env.example ──────────────────────────────────────────────────────────
cat > .env.example << 'ENVEXAMPLE'
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=c2-detection

# API
API_HOST=0.0.0.0
API_PORT=8000
ALERT_THRESHOLD=0.7

# Data paths
CTU13_DATA_DIR=data/raw/ctu13
PROCESSED_DATA_DIR=data/processed

# Model
MODEL_PATH=models/artifacts/graphsage_best.pt
WINDOW_SIZE_SECONDS=60
REALTIME_FACTOR=10.0
ENVEXAMPLE

# ─── 7. pre-commit config ─────────────────────────────────────────────────────
echo "→ Writing pre-commit config..."
cat > .pre-commit-config.yaml << 'PRECOMMIT'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: ["--maxkb=10240"]
      - id: check-merge-conflict
      - id: detect-private-key

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.7
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, pydantic]

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.8
    hooks:
      - id: bandit
        args: ["-c", "pyproject.toml"]
PRECOMMIT

# ─── 8. Makefile ──────────────────────────────────────────────────────────────
echo "→ Writing Makefile..."
cat > Makefile << 'MAKEFILE'
.PHONY: setup install lint test train-xgb train-graphsage train-gat demo docker-build docker-up mlflow clean help

# ──────────────────────────────────────────────────────────────────────────────
# Colors
RED    := \033[0;31m
GREEN  := \033[0;32m
YELLOW := \033[1;33m
NC     := \033[0m

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# ──────────────────────────────────────────────────────────────────────────────
setup: ## Initial project setup
	uv sync
	pre-commit install
	dvc init --no-scm || true
	@echo "$(GREEN)✓ Setup complete$(NC)"

install: ## Install production dependencies only
	uv sync --no-dev

lint: ## Run all linters
	ruff check src tests
	ruff format --check src tests
	mypy src
	bandit -r src -c pyproject.toml

format: ## Auto-format code
	ruff format src tests
	ruff check --fix src tests

test: ## Run tests with coverage
	pytest tests/ -v --cov=src/c2gnn --cov-report=term-missing

# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
demo: ## Run realtime simulation demo
	python -m c2gnn.realtime.pipeline \
		--data data/processed/ctu13_scenario10_flows.parquet \
		--model models/artifacts/graphsage_best.pt \
		--realtime-factor 20.0

api: ## Start FastAPI server
	uvicorn c2gnn.api:app --host 0.0.0.0 --port 8000 --reload

dashboard: ## Start Streamlit dashboard
	streamlit run src/c2gnn/dashboard.py

mlflow: ## Start MLflow UI
	mlflow ui --port 5000

# ──────────────────────────────────────────────────────────────────────────────
docker-build: ## Build Docker image
	docker build -t c2gnn:latest .

docker-up: ## Start all services via docker-compose
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

# ──────────────────────────────────────────────────────────────────────────────
security-scan: ## Run security scans
	bandit -r src -f json -o reports/bandit_report.json || true
	trivy fs . --exit-code 0 --format table

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov coverage.xml

MAKEFILE

# ─── 9. GitHub Actions: CI ────────────────────────────────────────────────────
echo "→ Writing GitHub Actions workflows..."
cat > .github/workflows/ci.yml << 'CI'
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11"]

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Set up Python ${{ matrix.python-version }}
        run: uv python install ${{ matrix.python-version }}

      - name: Install dependencies
        run: uv sync --dev

      - name: Lint with ruff
        run: uv run ruff check src tests

      - name: Format check
        run: uv run ruff format --check src tests

      - name: Type check with mypy
        run: uv run mypy src

      - name: Run tests
        run: uv run pytest tests/ --cov=src/c2gnn --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          fail_ci_if_error: false
CI

cat > .github/workflows/security.yml << 'SECURITY'
name: Security Scan

on:
  push:
    branches: [main]
  schedule:
    - cron: "0 9 * * 1"   # every Monday 9am

jobs:
  bandit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        uses: astral-sh/setup-uv@v4
      - name: Install dependencies
        run: uv sync --dev
      - name: Run Bandit
        run: uv run bandit -r src -c pyproject.toml -f json -o reports/bandit.json || true
      - name: Upload Bandit report
        uses: actions/upload-artifact@v4
        with:
          name: bandit-report
          path: reports/bandit.json

  trivy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: "fs"
          scan-ref: "."
          format: "sarif"
          output: "trivy-results.sarif"
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: "trivy-results.sarif"
SECURITY

# ─── 10. Dockerfile ───────────────────────────────────────────────────────────
cat > Dockerfile << 'DOCKERFILE'
# ─── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml .
COPY src/ src/

# Install dependencies (CPU-only torch for container)
RUN uv pip install --system --no-cache \
    torch==2.3.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --system --no-cache \
    torch-geometric \
    --extra-index-url https://data.pyg.org/whl/torch-2.3.0+cpu.html
RUN uv pip install --system --no-cache -e ".[dev]" --no-deps

# ─── Runtime stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Non-root user
RUN groupadd -r c2gnn && useradd -r -g c2gnn c2gnn

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src ./src

# Data and model volumes
RUN mkdir -p data/processed models/artifacts reports && \
    chown -R c2gnn:c2gnn /app

USER c2gnn

EXPOSE 8000

CMD ["uvicorn", "c2gnn.api:app", "--host", "0.0.0.0", "--port", "8000"]
DOCKERFILE

# ─── 11. docker-compose.yml ───────────────────────────────────────────────────
cat > docker-compose.yml << 'COMPOSE'
version: "3.9"

services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data:ro
      - ./models:/app/models:ro
    env_file: .env
    depends_on:
      - mlflow
    restart: unless-stopped

  dashboard:
    build: .
    command: streamlit run src/c2gnn/dashboard.py --server.port 8501
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data:ro
    env_file: .env
    restart: unless-stopped

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.13.0
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlflow
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow/mlflow.db
    restart: unless-stopped

volumes:
  mlflow-data:
COMPOSE

# ─── 12. DVC setup ────────────────────────────────────────────────────────────
echo "→ Initializing DVC..."
dvc init --no-scm 2>/dev/null || true

cat > data/README.md << 'DATAREADME'
# Data Directory

> ⚠️ Raw data is NOT committed to Git. Use DVC to pull data.

## Structure
- `raw/ctu13/` — CTU-13 original .binetflow files (DVC tracked)
- `interim/` — partially processed files
- `processed/` — final .parquet files for training

## Download CTU-13
```bash
# Scenario 10 (Neris botnet, recommended for training)
wget -P data/raw/ctu13/ https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-Malware-Capture-Botnet-42/capture20110818.pcap.netflow.labeled

# Scenario 8 (Rbot, for generalization test)
wget -P data/raw/ctu13/ https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-Malware-Capture-Botnet-44/capture20110818-2.pcap.netflow.labeled

# Scenario 1 (Neris, original)
wget -P data/raw/ctu13/ https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-Malware-Capture-Botnet-42/capture20110818.binetflow
```
DATAREADME

# ─── 13. Init __init__.py files ───────────────────────────────────────────────
for dir in src/c2gnn src/c2gnn/data src/c2gnn/graph src/c2gnn/models \
           src/c2gnn/realtime src/c2gnn/evaluation src/c2gnn/utils; do
  echo '"""c2gnn package."""' > "$dir/__init__.py"
done
touch tests/__init__.py

# ─── 14. Git setup ────────────────────────────────────────────────────────────
echo "→ Initializing Git..."
git init
git checkout -b main
git add -A
git commit -m "chore: initial project scaffold with uv, ci, and devtools"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  ✅ Project initialized successfully!            ║"
echo "║                                                  ║"
echo "║  Next steps:                                     ║"
echo "║  1. git remote add origin <your-repo-url>        ║"
echo "║  2. git push -u origin main                      ║"
echo "║  3. make setup                                   ║"
echo "║  4. Download CTU-13: see data/README.md          ║"
echo "║  5. make mlflow  (in separate terminal)          ║"
echo "╚══════════════════════════════════════════════════╝"
