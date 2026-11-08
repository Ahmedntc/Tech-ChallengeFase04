"""Gate de ingestão: valida cada lote contra o contrato antes de chegar ao modelo.

- Lote aprovado: segue adiante com os tipos garantidos pelo contrato.
- Lote reprovado: é copiado para a quarentena, gera relatório (JSON + HTML) e,
  pela linha de comando, o processo termina com código de saída 1.

Uso: ``python -m credit_monitoring.contracts.gate caminho/do/lote.csv [--labeled]``
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import warnings
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa

from credit_monitoring.config import load_settings, resolve_path
from credit_monitoring.contracts.report import write_reports
from credit_monitoring.contracts.schema import RULES, build_contract, build_warning_checks
from credit_monitoring.data.columns import ID_COLUMN

logger = logging.getLogger(__name__)

MAX_EXAMPLES = 5
COLUMN_PRESENCE_CHECKS = ("column_in_schema", "column_in_dataframe")
RULE_IN_CHECK_NAME = re.compile(r"^([RW]\d+)_")


@dataclass
class RuleResult:
    rule_id: str
    title: str
    description: str
    blocking: bool
    evaluated: bool
    passed: bool
    affected_rows: int
    examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ValidationResult:
    batch_name: str
    validated_at: str
    n_rows: int
    passed: bool
    n_rows_with_violations: int
    rules: list[RuleResult]
    data: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def violated_rules(self) -> list[str]:
        return [r.rule_id for r in self.rules if r.blocking and not r.passed]

    @property
    def warnings(self) -> list[str]:
        return [r.rule_id for r in self.rules if not r.blocking and r.evaluated and not r.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_name": self.batch_name,
            "validated_at": self.validated_at,
            "status": "aprovado" if self.passed else "bloqueado",
            "n_rows": self.n_rows,
            "n_rows_with_violations": self.n_rows_with_violations,
            "violated_rules": self.violated_rules,
            "warnings": self.warnings,
            "rules": [asdict(rule) for rule in self.rules],
        }


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def classify_failure(check: Any, failure_case: Any) -> str:
    """Associa uma falha reportada pelo Pandera à regra do contrato correspondente."""
    check = str(check)
    is_type_check = check.startswith(("dtype(", "coerce_dtype("))
    # Um nulo também falha na checagem de tipo; contamos só uma vez, como nulo.
    if check == "not_nullable" or (is_type_check and _is_missing(failure_case)):
        return "R2"
    named = RULE_IN_CHECK_NAME.match(check)
    if named:
        return named.group(1)
    if check == "field_uniqueness":
        return "R7"
    # Demais falhas são estruturais: coluna ausente ou extra, ou tipo inválido.
    return "R1"


def _format_value(value: Any) -> str:
    return "nulo" if _is_missing(value) else str(value)


def _summarize(failure_cases: pd.DataFrame) -> dict[str, tuple[int, list[dict[str, Any]]]]:
    """Agrupa as falhas por regra: registros afetados e alguns exemplos."""
    if failure_cases.empty:
        return {}
    records = failure_cases.to_dict("records")
    for record in records:
        record["rule_id"] = classify_failure(record["check"], record["failure_case"])
        # Em falhas de presença de coluna, o Pandera põe o nome do schema no campo "column".
        if str(record["check"]) in COLUMN_PRESENCE_CHECKS:
            record["column"] = None
    frame = pd.DataFrame(records)

    summary: dict[str, tuple[int, list[dict[str, Any]]]] = {}
    for rule_id, group in frame.groupby("rule_id"):
        no_index = group[group["index"].isna()]
        # Um exemplo por registro; em falhas de registro inteiro, mostra o client_id.
        indexed = (
            group[group["index"].notna()]
            .sort_values("column", key=lambda col: col.ne(ID_COLUMN), kind="stable")
            .drop_duplicates("index")
            .sort_values("index")
        )
        examples = [
            {
                "indice": None if _is_missing(row["index"]) else int(row["index"]),
                "coluna": None if _is_missing(row["column"]) else str(row["column"]),
                "valor": _format_value(row["failure_case"]),
                "checagem": str(row["check"]),
            }
            for row in pd.concat([no_index, indexed]).head(MAX_EXAMPLES).to_dict("records")
        ]
        summary[str(rule_id)] = (int(indexed["index"].nunique()), examples)
    return summary


def _run_schema(
    schema: pa.DataFrameSchema, df: pd.DataFrame
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    """Valida em modo lazy (coleta todas as falhas). Devolve (dados validados, falhas)."""
    with warnings.catch_warnings():
        # Aviso interno do Pandera ao montar a tabela de falhas; não afeta o resultado.
        warnings.filterwarnings("ignore", category=FutureWarning, module="pandera")
        try:
            return schema.validate(df, lazy=True), pd.DataFrame()
        except pa.errors.SchemaErrors as exc:
            return None, exc.failure_cases


def validate_batch(df: pd.DataFrame, batch_name: str, labeled: bool = False) -> ValidationResult:
    """Aplica o contrato rígido e, se aprovado, as regras de alerta."""
    validated, failures = _run_schema(build_contract(labeled), df)
    passed = validated is not None
    hard = _summarize(failures)
    soft: dict[str, tuple[int, list[dict[str, Any]]]] = {}
    if passed:
        _, warning_failures = _run_schema(build_warning_checks(), validated)
        soft = _summarize(warning_failures)

    rules = []
    for rule in RULES.values():
        source = hard if rule.blocking else soft
        affected, examples = source.get(rule.rule_id, (0, []))
        rules.append(
            RuleResult(
                rule_id=rule.rule_id,
                title=rule.title,
                description=rule.description,
                blocking=rule.blocking,
                evaluated=rule.blocking or passed,
                passed=rule.rule_id not in source,
                affected_rows=affected,
                examples=examples,
            )
        )

    violating_rows = failures["index"].dropna().nunique() if not failures.empty else 0
    return ValidationResult(
        batch_name=batch_name,
        validated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        n_rows=len(df),
        passed=passed,
        n_rows_with_violations=int(violating_rows),
        rules=rules,
        data=validated,
    )


def read_batch(path: Path) -> pd.DataFrame:
    """Lê um lote recebido (CSV ou Parquet). O client_id é sempre lido como texto."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, dtype={ID_COLUMN: str})
    raise ValueError(f"Formato de lote não suportado: {path.suffix}")


def ingest_batch(
    path: Path, settings: dict[str, Any] | None = None, labeled: bool = False
) -> ValidationResult:
    """Valida um arquivo de lote, grava os relatórios e coloca em quarentena se reprovado."""
    settings = settings or load_settings()
    paths = settings["paths"]
    result = validate_batch(read_batch(path), path.stem, labeled)
    write_reports(result, resolve_path(paths["validation_reports_dir"]))

    if result.passed:
        logger.info("Lote %s aprovado (%d registros).", result.batch_name, result.n_rows)
    else:
        quarantine_dir = resolve_path(paths["quarantine_dir"])
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        shutil.copy2(path, quarantine_dir / f"{stamp}_{path.name}")
        logger.error(
            "Lote %s BLOQUEADO: regras violadas %s; enviado para quarentena.",
            result.batch_name,
            ", ".join(result.violated_rules),
        )
    return result


def format_summary(result: ValidationResult) -> str:
    """Resumo legível do resultado para o terminal."""
    status = "APROVADO" if result.passed else "BLOQUEADO"
    lines = [f"Lote: {result.batch_name} ({result.n_rows} registros) -> {status}"]
    for rule in result.rules:
        if not rule.evaluated:
            mark = "  -  "
        elif rule.passed:
            mark = "  ok "
        else:
            mark = " FALHA" if rule.blocking else " ALERTA"
        detail = f"{rule.affected_rows} registro(s)" if not rule.passed else ""
        lines.append(f"  [{mark.strip():^6}] {rule.rule_id} {rule.title:<26} {detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Valida um lote contra o contrato de dados.")
    parser.add_argument("batch", type=Path, help="Arquivo do lote (.csv ou .parquet)")
    parser.add_argument("--labeled", action="store_true", help="O lote contém o alvo (default)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = ingest_batch(args.batch, labeled=args.labeled)
    print(format_summary(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
