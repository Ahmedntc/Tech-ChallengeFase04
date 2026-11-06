import json

import numpy as np
import pandas as pd
import pytest

from credit_monitoring.contracts.corrupt import make_corrupted_batch
from credit_monitoring.contracts.gate import ingest_batch, validate_batch
from credit_monitoring.contracts.schema import LIMIT_BAL_MAX
from credit_monitoring.data.columns import ID_COLUMN, TARGET


def _set(column, value, row=0):
    def mutate(df):
        df = df.astype({column: object}) if isinstance(value, str) else df.copy()
        df.loc[row, column] = value
        return df

    return mutate


def _append_copy_of_first_row(df):
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


def _reuse_client_id(df):
    df = df.copy()
    df.loc[1, ID_COLUMN] = df.loc[0, ID_COLUMN]
    return df


VIOLATIONS = [
    ("R1", "coluna extra (dado pessoal vazado)", lambda df: df.assign(cpf="000.000.000-00")),
    ("R1", "coluna obrigatória ausente", lambda df: df.drop(columns="age")),
    ("R1", "texto em campo numérico", _set("age", "trinta")),
    ("R2", "limite nulo", _set("limit_bal", np.nan)),
    ("R2", "escolaridade nula", _set("education", np.nan)),
    ("R3", "menor de idade", _set("age", 17)),
    ("R3", "idade implausível", _set("age", 101)),
    ("R4", "limite zero", _set("limit_bal", 0.0)),
    ("R4", "limite acima do teto", _set("limit_bal", LIMIT_BAL_MAX + 1.0)),
    ("R5", "sexo inválido", _set("sex", 3)),
    ("R5", "escolaridade inválida", _set("education", 0)),
    ("R5", "estado civil inválido", _set("marriage", 4)),
    ("R5", "status de pagamento inválido", _set("pay_status_3", 10)),
    ("R6", "pagamento negativo", _set("pay_amt_6", -1.0)),
    ("R7", "registro reenviado", _append_copy_of_first_row),
    ("R7", "client_id repetido", _reuse_client_id),
]


def test_valid_batch_passes(valid_batch):
    result = validate_batch(valid_batch, "valido")
    assert result.passed
    assert result.violated_rules == []
    assert result.data is not None and len(result.data) == len(valid_batch)


@pytest.mark.parametrize(
    ("rule_id", "mutate"),
    [(rule, mutate) for rule, _, mutate in VIOLATIONS],
    ids=[name for _, name, _ in VIOLATIONS],
)
def test_each_violation_blocks_and_is_attributed_to_the_right_rule(valid_batch, rule_id, mutate):
    result = validate_batch(mutate(valid_batch), "invalido")
    assert not result.passed
    assert result.data is None
    assert result.violated_rules == [rule_id]


def test_invalid_text_does_not_hide_other_violations_in_the_same_column(valid_batch):
    df = valid_batch.astype({"age": object})
    df.loc[0, "age"] = "trinta"
    df.loc[1, "age"] = 15
    result = validate_batch(df, "misto")
    assert result.violated_rules == ["R1", "R3"]


def test_affected_rows_are_counted_per_rule(valid_batch):
    df = valid_batch.copy()
    df.loc[[0, 1, 2], "age"] = 10
    df.loc[3, "pay_amt_1"] = -50.0
    result = validate_batch(df, "contagem")
    affected = {r.rule_id: r.affected_rows for r in result.rules}
    assert affected["R3"] == 3
    assert affected["R6"] == 1
    assert result.n_rows_with_violations == 4


def test_labeled_contract_validates_the_target(valid_batch):
    assert validate_batch(valid_batch, "sem_alvo", labeled=True).violated_rules == ["R1"]
    labeled = valid_batch.assign(**{TARGET: 0})
    assert validate_batch(labeled, "com_alvo", labeled=True).passed
    labeled.loc[0, TARGET] = 2
    assert validate_batch(labeled, "alvo_invalido", labeled=True).violated_rules == ["R5"]


def test_warnings_flag_but_do_not_block(valid_batch):
    df = valid_batch.copy()
    df.loc[0, "bill_amt_3"] = -10.0
    df.loc[1, "bill_amt_1"] = df.loc[1, "limit_bal"] * 3
    result = validate_batch(df, "alertas")
    assert result.passed
    assert {"W1", "W2"} <= set(result.warnings)


def test_corrupted_batch_triggers_every_hard_rule(valid_batch):
    corrupted, manifest = make_corrupted_batch(valid_batch, seed=1)
    result = validate_batch(corrupted, "corrompido")
    assert not result.passed
    assert result.violated_rules == sorted(manifest)


def test_ingest_quarantines_blocked_batch_and_writes_reports(settings, valid_batch, tmp_path):
    corrupted, _ = make_corrupted_batch(valid_batch, seed=1)
    path = tmp_path / "lote_ruim.csv"
    corrupted.to_csv(path, index=False)

    result = ingest_batch(path, settings)

    assert not result.passed
    quarantined = list((tmp_path / "quarantine").glob("*_lote_ruim.csv"))
    assert len(quarantined) == 1
    report = json.loads((tmp_path / "reports" / "lote_ruim.json").read_text(encoding="utf-8"))
    assert report["status"] == "bloqueado"
    assert (tmp_path / "reports" / "lote_ruim.html").exists()


def test_ingest_accepts_valid_csv_without_quarantine(settings, valid_batch, tmp_path):
    path = tmp_path / "lote_bom.csv"
    valid_batch.to_csv(path, index=False)

    result = ingest_batch(path, settings)

    assert result.passed
    assert not (tmp_path / "quarantine").exists()
    assert result.data[ID_COLUMN].equals(valid_batch[ID_COLUMN])
