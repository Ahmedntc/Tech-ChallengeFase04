"""Log centralizado e estruturado (JSON Lines).

Todas as etapas do pipeline escrevem no mesmo arquivo (``logs/pipeline.jsonl``): uma
linha JSON por evento, sempre com o ``run_id`` da execução. Os eventos de negócio
(tarefa concluída, resumo mensal, alerta) levam campos estruturados, e é desse
arquivo que o dashboard de monitoramento é montado.

Nenhum dado pessoal é registrado: apenas contagens, métricas e nomes de arquivos.
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import UTC, datetime
from typing import Any

from credit_monitoring.config import load_settings, resolve_path

CONSOLE_FORMAT = "%(levelname)s %(name)s: %(message)s"
OWN_LOGGERS = ("credit_monitoring", "__main__")
_HANDLER_TAG = "_credit_monitoring_handler"

_run_id: str | None = None


def new_run_id() -> str:
    """Identificador único da execução: instante UTC + sufixo aleatório."""
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(2)}"


def current_run_id() -> str | None:
    return _run_id


class JsonFormatter(logging.Formatter):
    """Serializa cada registro de log como uma linha JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "run_id": _run_id,
            "logger": record.name,
            "event": getattr(record, "event", "log"),
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _OwnOrWarningFilter(logging.Filter):
    """Mantém os logs do projeto e, de bibliotecas de terceiros, só avisos e erros."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(OWN_LOGGERS) or record.levelno >= logging.WARNING


def configure_logging(settings: dict[str, Any] | None = None, run_id: str | None = None) -> str:
    """Liga o console (legível) e o arquivo central (JSONL). Devolve o ``run_id`` em uso.

    Pode ser chamada mais de uma vez: os handlers anteriores são substituídos.
    """
    global _run_id
    settings = settings or load_settings()
    reset_logging()
    _run_id = run_id or new_run_id()
    root = logging.getLogger()

    log_path = resolve_path(settings["paths"]["logs_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    central = logging.FileHandler(log_path, encoding="utf-8")
    central.setFormatter(JsonFormatter())
    for handler in (console, central):
        setattr(handler, _HANDLER_TAG, True)
        handler.addFilter(_OwnOrWarningFilter())
        root.addHandler(handler)
    root.setLevel(logging.INFO)
    return _run_id


def log_event(logger: logging.Logger, level: int, event: str, message: str, **fields: Any) -> None:
    """Registra um evento estruturado: ``event`` identifica o tipo e ``fields`` os dados."""
    logger.log(level, message, extra={"event": event, "fields": fields})


def reset_logging() -> None:
    """Remove os handlers instalados por ``configure_logging`` e esquece o ``run_id``."""
    global _run_id
    _run_id = None
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, _HANDLER_TAG, False)]:
        root.removeHandler(handler)
        handler.close()
