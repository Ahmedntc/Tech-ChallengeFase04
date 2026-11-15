"""Orquestração do pipeline completo, com observabilidade de ponta a ponta.

Uma execução (identificada por um ``run_id``) percorre:

1. **Preparação:** verificar o dataset, preparar os dados, treinar o modelo e simular
   os meses de produção.
2. **Monitoramento:** para cada mês, ingestão pelo contrato, predição, avaliação de
   desempenho e detecção de drift, com alertas.
3. **Fechamento:** relatórios, registro no MLflow e dashboard.

Cada tarefa registra status, duração, volume e memória no log centralizado. O
dashboard é gerado mesmo quando a execução falha, para que a falha fique visível.

Uso: ``python -m credit_monitoring.pipeline [--no-mlflow]``
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from credit_monitoring.config import load_settings
from credit_monitoring.data.download import ensure_raw_dataset
from credit_monitoring.data.prepare import prepare
from credit_monitoring.drift.monitor import MonthResult, run_monitoring
from credit_monitoring.drift.simulate import simulate
from credit_monitoring.models.train import train
from credit_monitoring.monitoring.alerts import log_alert
from credit_monitoring.monitoring.dashboard import build_dashboard
from credit_monitoring.monitoring.telemetry import (
    finish_run,
    infrastructure_alerts,
    start_run,
    task_records,
    track_task,
)
from credit_monitoring.reporting import num

logger = logging.getLogger(__name__)

STAGE = "preparacao"


@dataclass
class PipelineResult:
    run_id: str
    status: str
    summary: dict[str, Any]
    months: list[MonthResult] = field(default_factory=list)
    dashboard: Path | None = None


def _run_stages(settings: dict[str, Any], run_id: str, track: bool) -> list[MonthResult]:
    with track_task("verificar_dataset", stage=STAGE) as task:
        raw = ensure_raw_dataset(settings)
        task.rows_in = task.rows_out = len(raw)

    with track_task("preparar_dados", stage=STAGE) as task:
        summary = prepare(settings)
        cleaning = summary["cleaning"]
        task.rows_in, task.rows_out = cleaning["rows_in"], cleaning["rows_out"]
        task.rows_rejected = cleaning["duplicates_removed"]
        task.detail = (
            f"{cleaning['duplicates_removed']} duplicatas removidas; "
            f"Referência com {num(summary['reference_rows'], 0)} e Pool de Produção com "
            f"{num(summary['production_pool_rows'], 0)} registros"
        )

    with track_task("treinar_modelo", stage=STAGE) as task:
        bundle, _ = train(settings, track=track)
        data = bundle.metadata["data"]
        task.rows_in = task.rows_out = data["n_train"] + data["n_valid"] + data["n_test"]
        task.detail = f"campeão {bundle.metadata['champion']}, versão {bundle.version}"

    with track_task("simular_producao", stage=STAGE) as task:
        manifest = simulate(settings)
        task.rows_out = sum(month["stats"]["rows"] for month in manifest["months"])
        task.detail = f"{len(manifest['months'])} meses"

    months = run_monitoring(settings)

    if track:
        with track_task("registrar_mlflow", stage="monitoramento") as task:
            from credit_monitoring.monitoring.tracking import log_monitoring_run

            log_monitoring_run(settings, run_id, bundle.version, months, task_records())
            task.detail = f"experimento {settings['mlflow']['monitoring_experiment']}"
    return months


def _close_run(
    settings: dict[str, Any], run_id: str, status: str, started: float, error: str | None
) -> tuple[dict[str, Any], Path]:
    """Encerra a execução: alertas de infraestrutura, resumo no log e dashboard."""
    for alert in infrastructure_alerts(task_records(), settings["observability"]):
        log_alert(logger, alert)
    summary = finish_run(status, time.perf_counter() - started, error)
    return summary, build_dashboard(settings, run_id)


def run_pipeline(settings: dict[str, Any] | None = None, track: bool = True) -> PipelineResult:
    """Executa o pipeline completo. ``track=False`` pula o registro no MLflow."""
    settings = settings or load_settings()
    run_id = start_run(settings)
    started = time.perf_counter()
    try:
        months = _run_stages(settings, run_id, track)
    except Exception as exc:
        logger.exception("Execução %s interrompida.", run_id)
        try:
            _close_run(settings, run_id, "falha", started, f"{type(exc).__name__}: {exc}")
        except Exception:
            # O erro original é o que importa; uma falha ao montar o dashboard não o esconde.
            logger.exception("Não foi possível gerar o dashboard da execução com falha.")
        raise
    summary, dashboard = _close_run(settings, run_id, "sucesso", started, None)
    return PipelineResult(run_id, "sucesso", summary, months, dashboard)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Executa o pipeline completo de monitoramento.")
    parser.add_argument("--no-mlflow", action="store_true", help="Não registra no MLflow")
    args = parser.parse_args(argv)

    settings = load_settings()
    try:
        result = run_pipeline(settings, track=not args.no_mlflow)
    except Exception:
        print(f"\nPipeline FALHOU. Veja o log em {settings['paths']['logs_file']}.")
        return 1

    s = result.summary
    print(f"\nExecução {result.run_id}: {result.status} em {s['duration_seconds']:.1f}s")
    print(
        f"Tarefas: {s['tasks']} ({s['tasks_failed']} falha(s), {s['tasks_blocked']} bloqueio(s)) | "
        f"registros processados: {s['rows_processed']} | rejeitados: {s['rows_rejected']} | "
        f"pico de memória: {s['memory_peak_mb']:.0f} MB"
    )
    print(
        f"\n{'mês':<5}{'drift':>7}{'AUC':>8}{'score':>8}{'real':>8}{'alertas':>9}{'críticos':>10}"
    )
    for r in result.months:
        critical = sum(a.severity == "crítico" for a in r.alerts)
        print(
            f"{r.month:<5}{r.n_drifted:>4}/23{r.performance['auc']:>8.3f}{r.mean_score:>8.1%}"
            f"{r.performance['actual_default_rate']:>8.1%}{len(r.alerts):>9}{critical:>10}"
        )
    print(f"\nDashboard: {settings['paths']['monitoring_reports_dir']}/dashboard.html")
    print(f"Relatório de drift: {settings['paths']['drift_reports_dir']}/drift_report.html")
    print(f"Log centralizado: {settings['paths']['logs_file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
