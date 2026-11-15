"""Alertas do monitoramento: estrutura comum e registro no log centralizado."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from credit_monitoring.monitoring.logger import log_event

# Severidade do alerta → nível do log.
ALERT_LEVELS = {"crítico": logging.ERROR, "atenção": logging.WARNING, "info": logging.INFO}


@dataclass
class Alert:
    month: str
    category: str
    severity: str
    subject: str
    message: str
    value: float | None = None
    threshold: float | None = None


def br(value: float, spec: str) -> str:
    """Formata números no padrão brasileiro (vírgula decimal) para as mensagens."""
    return format(value, spec).replace(".", ",")


def log_alert(logger: logging.Logger, alert: Alert) -> None:
    fields = asdict(alert)
    message = fields.pop("message")
    log_event(logger, ALERT_LEVELS[alert.severity], "alerta", message, **fields)
