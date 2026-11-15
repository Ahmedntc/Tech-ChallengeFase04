import copy
import json

from credit_monitoring.governance.report import build, save


def test_governance_report_is_built_from_a_pipeline_run(pipeline_run):
    cfg, tmp, _, _ = pipeline_run
    cfg = copy.deepcopy(cfg)
    gov = cfg["governance"]
    gov["causal"].update(month="M5", seeds=2)
    gov["retraining"]["repetitions"] = 1
    gov["fairness"]["seeds"] = [1]
    gov["fairness"]["production_windows"] = {"Início": ["M1"], "Crise": ["M4", "M5"]}
    gov["retention"] = {str(tmp / "quarantine"): 90}

    result = build(cfg)
    path = save(result)

    html = path.read_text(encoding="utf-8")
    for section in ("Análise causal", "Prescrição", "Equidade", "Explicabilidade", "Privacidade"):
        assert section in html
    for chart in ("chart-causal-auc", "chart-causal-gap", "chart-importance"):
        assert f'id="{chart}"' in html

    out = tmp / "reports" / "governance"
    for name in (
        "causal_by_mechanism.csv",
        "retraining.csv",
        "fairness_indicators.csv",
        "fairness_variants.csv",
        "fairness_production.csv",
        "reason_codes_examples.csv",
        "reidentification_risk.csv",
    ):
        assert (out / name).exists(), name
    summary = json.loads((out / "causal_summary.json").read_text(encoding="utf-8"))
    assert summary["month"] == "M5"
    assert set(result["production"]["window"]) == {"Início", "Crise"}
    assert len(result["reasons"]) == len(result["examples"]) > 0
