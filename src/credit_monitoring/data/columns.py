"""Nomes de colunas padronizados, compartilhados por todo o pipeline."""

ID_COLUMN = "client_id"
TARGET = "default"

CATEGORICAL = ["sex", "education", "marriage"]
# Índice 1 = setembro/2005 (mais recente) ... 6 = abril/2005.
PAY_STATUS = [f"pay_status_{i}" for i in range(1, 7)]
BILL_AMT = [f"bill_amt_{i}" for i in range(1, 7)]
PAY_AMT = [f"pay_amt_{i}" for i in range(1, 7)]

MONETARY = ["limit_bal", *BILL_AMT, *PAY_AMT]
INTEGER = [*CATEGORICAL, "age", *PAY_STATUS]
FEATURES = ["limit_bal", *CATEGORICAL, "age", *PAY_STATUS, *BILL_AMT, *PAY_AMT]

# Mapeamento das colunas originais da UCI para os nomes padronizados.
RAW_COLUMN_MAP = {
    "ID": "id",
    "LIMIT_BAL": "limit_bal",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "pay_status_1",
    **{f"PAY_{i}": f"pay_status_{i}" for i in range(2, 7)},
    **{f"BILL_AMT{i}": f"bill_amt_{i}" for i in range(1, 7)},
    **{f"PAY_AMT{i}": f"pay_amt_{i}" for i in range(1, 7)},
    "default payment next month": TARGET,
}
