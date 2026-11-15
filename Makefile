# Atalhos do projeto. Requer Python >= 3.12.
PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python

.DEFAULT_GOAL := help
.PHONY: help setup data prepare contract-demo validate train simulate drift pipeline dashboard \
	governance explain retention mlflow-ui all lint format test clean

help: ## Lista os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

setup: ## Cria o ambiente virtual e instala as dependências
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps

data: ## Garante o dataset bruto e verifica sua integridade
	$(PY) -m credit_monitoring.data.download

prepare: ## Limpa, pseudonimiza e separa Referência x Pool de Produção
	$(PY) -m credit_monitoring.data.prepare

contract-demo: ## Mostra o contrato aprovando um lote válido e bloqueando um corrompido
	$(PY) -m credit_monitoring.contracts.demo

validate: ## Valida um lote qualquer: make validate BATCH=caminho/do/lote.csv
	@test -n "$(BATCH)" || (echo "Informe o lote: make validate BATCH=caminho/do/lote.csv" && exit 2)
	$(PY) -m credit_monitoring.contracts.gate $(BATCH)

train: ## Treina os modelos baseline e registra no MLflow
	$(PY) -m credit_monitoring.models.train

simulate: ## Simula 6 meses de produção com drift crescente
	$(PY) -m credit_monitoring.drift.simulate

drift: ## Monitora os meses simulados: contrato, predição, drift, desempenho e alertas
	$(PY) -m credit_monitoring.drift.monitor

pipeline: ## Executa tudo de ponta a ponta com observabilidade (log central, MLflow e dashboard)
	$(PY) -m credit_monitoring.pipeline

dashboard: ## Regera o dashboard a partir do log centralizado
	$(PY) -m credit_monitoring.monitoring.dashboard

governance: ## Análise causal, equidade, explicabilidade e privacidade (relatório de governança)
	$(PY) -m credit_monitoring.governance.report

explain: ## Explica a decisão de um cliente: make explain BATCH=lote.csv CLIENT=<client_id>
	@test -n "$(BATCH)" -a -n "$(CLIENT)" || (echo "Uso: make explain BATCH=data/incoming/M6.csv CLIENT=<client_id>" && exit 2)
	$(PY) -m credit_monitoring.governance.explain $(BATCH) $(CLIENT)

retention: ## Lista os arquivos com prazo de retenção vencido (APPLY=1 para apagar)
	$(PY) -m credit_monitoring.governance.privacy $(if $(APPLY),--apply,)

mlflow-ui: ## Abre a interface do MLflow em http://127.0.0.1:5000
	MLFLOW_DISABLE_TELEMETRY=true $(BIN)/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

all: pipeline contract-demo governance ## Pipeline completo + demonstração do contrato + governança

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
