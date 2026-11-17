# Imagem única do projeto: pipeline, testes e interface do MLflow.
# Uso recomendado via docker compose (ver docker-compose.yml e "make help").
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# make: os comandos do projeto são alvos do Makefile.
# git: o MLflow registra em cada execução o commit do código que a gerou (rastreabilidade). A pasta do
# projeto é montada em /app e pode pertencer a outro usuário, então ela é marcada como segura.
# (O XGBoost e o scikit-learn trazem a própria libgomp nos wheels: não há outras bibliotecas a instalar.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends make git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /app

# O Makefile usa o ambiente virtual indicado em VENV (fora da pasta do projeto, que é montada
# como volume). MLflow sem telemetria: nada sai do ambiente.
ENV VENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLFLOW_DISABLE_TELEMETRY=true \
    MLFLOW_DISABLE_AGENT_HINT=1

RUN python -m venv /opt/venv

WORKDIR /app

# Dependências primeiro: a camada só é refeita quando os requirements mudam.
# A NCCL (cerca de 350 MB) só serve para treino distribuído em GPU; o XGBoost a carrega sob demanda
# e o projeto treina em CPU, então ela sai da imagem.
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt \
    && (pip uninstall -y nvidia-nccl-cu12 nvidia-nccl-cu13 || true)

# Código, configuração e dataset bruto (o .dockerignore deixa de fora segredos e arquivos gerados).
COPY . .
RUN pip install -e . --no-deps

# Usuário sem privilégios. No docker compose, o container roda com o UID do usuário da máquina,
# para que os arquivos gerados na pasta do projeto pertençam a ele.
RUN useradd --uid 1000 --create-home app && chown -R app:app /app
USER app
ENV USER=app

CMD ["make", "all"]
