"""Relatórios de validação do contrato de dados (JSON + HTML autocontido)."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from credit_monitoring.contracts.gate import RuleResult, ValidationResult

STRUCTURAL_LABELS = {
    "column_in_schema": "coluna não prevista no contrato",
    "column_in_dataframe": "coluna obrigatória ausente",
}

STYLE = """
:root { --bg:#f6f7f9; --card:#fff; --text:#1d2330; --muted:#5b6474; --line:#e3e6eb;
        --ok:#1f7a4d; --ok-bg:#e3f4ea; --fail:#b42318; --fail-bg:#fde8e7;
        --warn:#8a5a00; --warn-bg:#fff4d6; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
main { max-width:1100px; margin:0 auto; padding:32px 16px 48px; }
.eyebrow { margin:0; color:var(--muted); font-size:13px; text-transform:uppercase; letter-spacing:.06em; }
h1 { margin:4px 0 12px; font-size:28px; }
.badge { display:inline-block; padding:4px 12px; border-radius:999px; font-weight:600; font-size:14px; }
.badge.ok { background:var(--ok-bg); color:var(--ok); }
.badge.fail { background:var(--fail-bg); color:var(--fail); }
.stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:24px 0; }
.stat { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
.stat b { display:block; font-size:24px; font-variant-numeric:tabular-nums; }
.stat span { color:var(--muted); font-size:13px; }
.table-wrap { overflow-x:auto; background:var(--card); border:1px solid var(--line); border-radius:10px; }
table { width:100%; border-collapse:collapse; min-width:760px; }
th, td { padding:10px 12px; text-align:left; vertical-align:top; border-bottom:1px solid var(--line); }
th { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
tr:last-child td { border-bottom:none; }
td.num { font-variant-numeric:tabular-nums; text-align:right; }
.desc { color:var(--muted); font-size:13px; }
.pill { display:inline-block; white-space:nowrap; padding:2px 8px; border-radius:6px; font-size:12px; font-weight:600; }
.pill.ok { background:var(--ok-bg); color:var(--ok); }
.pill.fail { background:var(--fail-bg); color:var(--fail); }
.pill.warn { background:var(--warn-bg); color:var(--warn); }
.pill.skip { background:var(--line); color:var(--muted); }
td.ex-cell { min-width:300px; }
ul.ex { margin:0; padding-left:18px; font-size:13px; }
code { white-space:nowrap; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }
footer { margin-top:24px; color:var(--muted); font-size:13px; }
"""


def _status_pill(rule: RuleResult) -> str:
    if not rule.evaluated:
        return '<span class="pill skip">não avaliada</span>'
    if rule.passed:
        return '<span class="pill ok">ok</span>'
    if rule.blocking:
        return '<span class="pill fail">violada</span>'
    return '<span class="pill warn">alerta</span>'


def _example_text(example: dict[str, Any]) -> str:
    check = example["checagem"]
    if example["indice"] is None:
        label = STRUCTURAL_LABELS.get(check, check)
        return f"{escape(label)}: <code>{escape(example['valor'])}</code>"
    column = example["coluna"] or "registro"
    return (
        f"índice {example['indice']} · <code>{escape(column)}</code> = "
        f"<code>{escape(example['valor'])}</code>"
    )


def render_html(result: ValidationResult) -> str:
    badge = (
        '<span class="badge ok">APROVADO</span>'
        if result.passed
        else '<span class="badge fail">BLOQUEADO</span>'
    )
    rows = []
    for rule in result.rules:
        examples = "".join(f"<li>{_example_text(ex)}</li>" for ex in rule.examples)
        rows.append(
            "<tr>"
            f"<td><b>{rule.rule_id}</b></td>"
            f"<td><b>{escape(rule.title)}</b><div class='desc'>{escape(rule.description)}</div></td>"
            f"<td>{'rígida' if rule.blocking else 'alerta'}</td>"
            f"<td>{_status_pill(rule)}</td>"
            f"<td class='num'>{rule.affected_rows if rule.evaluated else '—'}</td>"
            f"<td class='ex-cell'>{f'<ul class=ex>{examples}</ul>' if examples else ''}</td>"
            "</tr>"
        )
    stats = [
        (result.n_rows, "registros recebidos"),
        (result.n_rows_with_violations, "registros com violação"),
        (len(result.violated_rules), "regras rígidas violadas"),
        (len(result.warnings), "alertas"),
    ]
    stats_html = "".join(
        f'<div class="stat"><b>{v}</b><span>{label}</span></div>' for v, label in stats
    )
    decision = (
        "Lote liberado para o modelo."
        if result.passed
        else "Lote bloqueado e enviado para a quarentena: nenhum registro chega ao modelo."
    )
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Validação — {escape(result.batch_name)}</title>
<style>{STYLE}</style>
</head>
<body>
<main>
  <p class="eyebrow">Relatório de validação do contrato de dados</p>
  <h1>{escape(result.batch_name)}</h1>
  {badge} <span class="desc">{escape(decision)}</span>
  <section class="stats">{stats_html}</section>
  <div class="table-wrap">
    <table>
      <thead><tr><th>ID</th><th>Regra</th><th>Tipo</th><th>Resultado</th>
      <th>Registros afetados</th><th>Exemplos</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>
  <footer>Validado em {escape(result.validated_at)} (UTC) · contrato Pandera ·
  regras de alerta só são avaliadas em lotes aprovados.</footer>
</main>
</body>
</html>
"""


def write_reports(result: ValidationResult, directory: Path) -> tuple[Path, Path]:
    """Grava o relatório em JSON (máquina) e HTML (pessoas)."""
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{result.batch_name}.json"
    html_path = directory / f"{result.batch_name}.html"
    json_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    html_path.write_text(render_html(result), encoding="utf-8")
    return json_path, html_path
