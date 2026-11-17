# Atalhos do projeto. Dois modos de execução:
#   - Docker (recomendado): make docker-build e depois make docker-<alvo>, sem instalar Python;
#   - ambiente local: make setup (requer Python >= 3.12) e depois make <alvo>.
# Dentro do container, VENV aponta para o ambiente da imagem (/opt/venv).
VENV ?= .venv

# No Windows, o Python se chama "python" e os executáveis do ambiente virtual ficam em Scripts/.
ifeq ($(OS),Windows_NT)
PYTHON ?= python
BIN    := $(VENV)/Scripts
else
PYTHON ?= python3
BIN    := $(VENV)/bin
endif
PY := $(BIN)/python

.DEFAULT_GOAL := help
.PHONY: help setup check-venv data prepare contract-demo validate train simulate drift pipeline \
	dashboard governance explain retention mlflow-ui all lint format test clean \
	docker-build docker-setup docker-shell docker-mlflow-ui

help: ## Lista os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'
	@printf "  %-18s %s\n" "docker-<alvo>" "Roda qualquer alvo acima no container (ex.: make docker-all, make docker-test)"

setup: ## Cria o ambiente virtual e instala as dependências (PYTHON=python3.12 para escolher a versão)
	@$(PYTHON) -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" || ( \
		echo "O projeto requer Python 3.12 ou superior. Encontrado: $$($(PYTHON) --version 2>&1)."; \
		echo "Instale o Python 3.12 e rode: make setup PYTHON=python3.12"; exit 1)
	$(PYTHON) -m venv --clear $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps

# Os demais comandos usam o Python do ambiente virtual: sem ele, orienta a rodar o setup.
check-venv:
	$(if $(wildcard $(PY) $(PY).exe),,$(error Ambiente virtual não encontrado em $(VENV)/. Rode primeiro: make setup))

data prepare contract-demo validate train simulate drift pipeline dashboard governance explain \
retention mlflow-ui lint format test: check-venv

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
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} +

# --- Docker ---------------------------------------------------------------------------------------
# O container roda com o seu usuário, para que os arquivos gerados na pasta do projeto sejam seus.
# Os parâmetros dos alvos (BATCH, CLIENT, APPLY) são repassados para dentro do container.
DOCKER_COMPOSE ?= docker compose
PYTHON_VERSION ?= 3.12
DOCKER_ENV := HOST_UID=$(shell id -u 2>/dev/null) HOST_GID=$(shell id -g 2>/dev/null) PYTHON_VERSION=$(PYTHON_VERSION)
DOCKER_RUN := $(DOCKER_ENV) $(DOCKER_COMPOSE) run --rm app
DOCKER_ARGS = $(if $(BATCH),BATCH=$(BATCH)) $(if $(CLIENT),CLIENT=$(CLIENT)) $(if $(APPLY),APPLY=$(APPLY))

docker-build: ## Constrói a imagem Docker (PYTHON_VERSION=3.13 para outra versão do Python)
	$(DOCKER_ENV) $(DOCKER_COMPOSE) build app

docker-shell: ## Abre um terminal dentro do container
	$(DOCKER_RUN) bash

docker-mlflow-ui: ## Abre a interface do MLflow (Docker) em http://127.0.0.1:5000
	$(DOCKER_ENV) $(DOCKER_COMPOSE) up mlflow

docker-setup:
	@echo "No Docker não há setup: as dependências já estão na imagem. Rode: make docker-build"

docker-%:
	$(DOCKER_RUN) make $* $(DOCKER_ARGS)
