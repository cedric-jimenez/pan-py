.PHONY: help install install-dev format lint type-check test clean run docker-build docker-run generate-openapi

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
VENV := venv
APP_MODULE := app.main:app

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

venv: ## Create virtual environment
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created. Activate it with: source $(VENV)/bin/activate"

format: ## Format code with Black
	black app/
	ruff check --fix app/

lint: ## Run linting checks (Ruff)
	ruff check app/
	black --check app/

type-check: ## Run type checking (mypy)
	mypy app/

check: lint type-check ## Run all checks (lint + type-check)

test: ## Run tests with pytest
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=app --cov-report=html --cov-report=term

run: ## Run the FastAPI development server
	uvicorn $(APP_MODULE) --reload --host 0.0.0.0 --port 8000 --log-config app/uvicorn_logging_config.json

run-prod: ## Run the FastAPI production server
	uvicorn $(APP_MODULE) --host 0.0.0.0 --port 8000 --log-config app/uvicorn_logging_config.json

docker-build: ## Build Docker image
	docker build -t salamander-detection-api:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 --name salamander-api salamander-detection-api:latest

docker-stop: ## Stop and remove Docker container
	docker stop salamander-api || true
	docker rm salamander-api || true

pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

generate-openapi: ## Generate OpenAPI specification from FastAPI app
	$(PYTHON) scripts/generate_openapi.py

clean: ## Clean up cache and build files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

.DEFAULT_GOAL := help
