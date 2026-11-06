# Dados

## Fonte

**Default of Credit Card Clients** — UCI Machine Learning Repository (id 350).

- Citação: Yeh, I-Cheng (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
- Artigo de referência: Yeh, I-C. & Lien, C-H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*. Expert Systems with Applications, 36(2), 2473–2480.
- Licença: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). A redistribuição é permitida com atribuição, por isso o arquivo original fica versionado neste repositório.
- Contexto: clientes de cartão de crédito em Taiwan, de abril a setembro de 2005.

| Arquivo | SHA-256 do arquivo |
|---|---|
| `raw/default_of_credit_card_clients.xls` | `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933` |

O pipeline confere a **impressão digital do conteúdo** (SHA-256 do CSV canônico) registrada em
`config/settings.yaml`, para detectar qualquer alteração nos dados independentemente do formato do arquivo.

## Camadas

| Pasta | Conteúdo | Versionado |
|---|---|---|
| `raw/` | Planilha original da UCI, imutável | sim |
| `reference/` | Dataset de Referência, o único usado no treino | não (gerado) |
| `production/` | Lotes mensais simulados onde o drift acontece | não (gerado) |
| `quarantine/` | Lotes rejeitados pelo contrato de dados | não (gerado) |

## Dicionário de dados (original)

30.000 clientes × 25 colunas. Valores monetários em novos dólares taiwaneses (NT$).

| Coluna | Descrição |
|---|---|
| `ID` | Identificador do cliente |
| `LIMIT_BAL` | Limite de crédito concedido (individual + familiar/suplementar) |
| `SEX` | 1 = masculino, 2 = feminino |
| `EDUCATION` | 1 = pós-graduação, 2 = universidade, 3 = ensino médio, 4 = outros, 5 e 6 = desconhecido |
| `MARRIAGE` | 1 = casado, 2 = solteiro, 3 = outros |
| `AGE` | Idade em anos |
| `PAY_0`, `PAY_2` … `PAY_6` | Status de pagamento de setembro (`PAY_0`) a abril (`PAY_6`): -1 = pago em dia, 1 a 9 = meses de atraso. Os códigos -2 e 0 não são documentados na fonte |
| `BILL_AMT1` … `BILL_AMT6` | Valor da fatura de setembro (`1`) a abril (`6`) |
| `PAY_AMT1` … `PAY_AMT6` | Valor pago de setembro (`1`) a abril (`6`) |
| `default payment next month` | **Alvo:** 1 = inadimplente em outubro/2005, 0 = adimplente |

## Problemas de qualidade encontrados no dado bruto

Esses pontos serão tratados (e documentados) na preparação do dataset de Referência:

| Problema | Ocorrências |
|---|---|
| Linhas duplicadas (ignorando `ID`) | 35 |
| `EDUCATION` com códigos não documentados (0) ou desconhecidos (5, 6) | 14 + 280 + 51 |
| `MARRIAGE` com código não documentado (0) | 54 |
| Clientes com alguma fatura negativa (saldo credor) | 1.930 |
| Distribuição do alvo | 22,12% inadimplentes (6.636 de 30.000) |
