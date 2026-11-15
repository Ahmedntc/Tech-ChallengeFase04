"""Telemetria das tarefas do pipeline: status, latência, volume e uso de recursos.

Cada tarefa roda dentro de ``track_task``, que mede duração, tempo de CPU e memória
do processo, guarda o resultado e o registra no log centralizado (evento
``task_finished``). Uma exceção marca a tarefa como ``falha`` e segue adiante para
quem chamou decidir o que fazer.

Status possíveis: ``sucesso``, ``bloqueado`` (o contrato de dados barrou o lote) e
``falha`` (erro técnico).
"""

from __future__ import annotations

import logging
import platform
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import psutil

from credit_monitoring.monitoring.alerts import Alert, br
from credit_monitoring.monitoring.logger import configure_logging, log_event

logger = logging.getLogger(__name__)

STATUS_LEVELS = {"sucesso": logging.INFO, "bloqueado": logging.WARNING, "falha": logging.ERROR}
MB = 1024 * 1024


@dataclass
class TaskRecord:
    task: str
    stage: str
    month: str | None = None
    batch: str | None = None
    status: str = "sucesso"
    started_at: str = ""
    duration_seconds: float = 0.0
    cpu_seconds: float = 0.0
    memory_rss_mb: float = 0.0
    rows_in: int | None = None
    rows_out: int | None = None
    rows_rejected: int = 0
    detail: str = ""
    error: str | None = None


_records: list[TaskRecord] = []


def task_records() -> list[TaskRecord]:
    """Tarefas registradas desde o início da execução atual."""
    return list(_records)


def reset_records() -> None:
    _records.clear()


def _cpu_seconds(process: psutil.Process) -> float:
    times = process.cpu_times()
    return times.user + times.system


@contextmanager
def track_task(
    task: str, *, stage: str, month: str | None = None, batch: str | None = None
) -> Iterator[TaskRecord]:
    """Mede e registra uma tarefa. O chamador preenche volumes e status no registro."""
    process = psutil.Process()
    record = TaskRecord(
        task=task,
        stage=stage,
        month=month,
        batch=batch,
        started_at=datetime.now(UTC).isoformat(timespec="milliseconds"),
    )
    cpu_start = _cpu_seconds(process)
    start = time.perf_counter()
    try:
        yield record
    except Exception as exc:
        record.status = "falha"
        record.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record.duration_seconds = round(time.perf_counter() - start, 4)
        record.cpu_seconds = round(_cpu_seconds(process) - cpu_start, 4)
        record.memory_rss_mb = round(process.memory_info().rss / MB, 1)
        _records.append(record)
        where = " · ".join(part for part in (month, batch) if part)
        log_event(
            logger,
            STATUS_LEVELS[record.status],
            "task_finished",
            f"Tarefa {task}{f' ({where})' if where else ''}: {record.status} "
            f"em {record.duration_seconds:.2f}s",
            **asdict(record),
        )


def start_run(settings: dict[str, Any]) -> str:
    """Inicia uma execução do pipeline: novo ``run_id``, log configurado e telemetria zerada."""
    run_id = configure_logging(settings)
    reset_records()
    log_event(
        logger,
        logging.INFO,
        "run_started",
        f"Execução {run_id} iniciada.",
        python=platform.python_version(),
        cpu_count=psutil.cpu_count(),
        memory_total_mb=round(psutil.virtual_memory().total / MB),
    )
    return run_id


def finish_run(status: str, duration_seconds: float, error: str | None = None) -> dict[str, Any]:
    """Encerra a execução e registra o resumo operacional (evento ``run_finished``)."""
    records = task_records()
    summary = {
        "status": status,
        "duration_seconds": round(duration_seconds, 2),
        "tasks": len(records),
        "tasks_failed": sum(r.status == "falha" for r in records),
        "tasks_blocked": sum(r.status == "bloqueado" for r in records),
        "rows_processed": sum(r.rows_out or 0 for r in records if r.task == "ingestao"),
        "rows_rejected": sum(r.rows_rejected for r in records if r.task == "ingestao"),
        "memory_peak_mb": max((r.memory_rss_mb for r in records), default=0.0),
        "error": error,
    }
    log_event(
        logger,
        logging.INFO if status == "sucesso" else logging.ERROR,
        "run_finished",
        f"Execução encerrada: {status} em {summary['duration_seconds']:.1f}s.",
        **summary,
    )
    return summary


def infrastructure_alerts(records: list[TaskRecord], cfg: dict[str, Any]) -> list[Alert]:
    """Alertas de estabilidade: tarefas lentas ou com memória acima do limite."""
    alerts = []
    for record in records:
        where = record.month or "execução"
        if record.duration_seconds > cfg["max_task_seconds"]:
            alerts.append(
                Alert(
                    where,
                    "infraestrutura",
                    "atenção",
                    record.task,
                    f"Tarefa {record.task} levou {br(record.duration_seconds, '.1f')}s "
                    f"(limite {cfg['max_task_seconds']}s).",
                    record.duration_seconds,
                    cfg["max_task_seconds"],
                )
            )
        if record.memory_rss_mb > cfg["max_memory_mb"]:
            alerts.append(
                Alert(
                    where,
                    "infraestrutura",
                    "atenção",
                    record.task,
                    f"Memória do processo em {record.memory_rss_mb:.0f} MB na tarefa "
                    f"{record.task} (limite {cfg['max_memory_mb']} MB).",
                    record.memory_rss_mb,
                    cfg["max_memory_mb"],
                )
            )
    return alerts
