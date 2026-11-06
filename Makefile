# Atalhos do projeto. Requer Python >= 3.12.
PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python

.DEFAULT_GOAL := help
.PHONY: help setup data lint format test clean

help: ## Lista os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

setup: ## Cria o ambiente virtual e instala as dependências
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps

data: ## Garante o dataset bruto e verifica sua integridade
	$(PY) -m credit_monitoring.data.download

lint: ## Verifica estilo e erros com ruff
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

format: ## Formata o código com ruff
	$(BIN)/ruff check --fix src tests
	$(BIN)/ruff format src tests

test: ## Roda a suíte de testes
	$(PY) -m pytest

clean: ## Remove caches (mantém o dataset bruto)
	rm -rf .pytest_cache .ruff_cache
	find . -path ./$(VENV) -prune -o -type d -name __pycache__ -exec rm -rf {} +
