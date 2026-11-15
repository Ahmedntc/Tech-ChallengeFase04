# Tech Challenge Fase 4 — Sustentação e Confiabilidade de um Modelo de Credit Scoring

> **Tema:** Observabilidade Contínua, Detecção de Drift, Qualidade de Dados e Governança em Produção

Uma fintech de crédito tem um modelo de *Credit Scoring* em produção. Com mudanças recentes na economia
(inflação, novos perfis de clientes), há suspeita de que o modelo esteja sofrendo **degradação silenciosa**.
Este projeto constrói a camada de sustentação e confiabilidade desse modelo:

1. **Contratos de dados** que impedem a entrada de dados ruins no modelo (Pandera).
2. **Simulação e detecção de drift** de dados e de conceito ao longo do tempo (Evidently AI, PSI, KS).
3. **Observabilidade** do pipeline, da infraestrutura e do modelo (logs estruturados, MLflow, dashboard).
4. **Governança e LGPD**, com análise causal da degradação.

## Status

| Etapa | Descrição | Status |
|---|---|---|
| 0 | Fundação do projeto (estrutura, dependências, ingestão do dataset) | concluída |
| 1 | Validação de dados e contratos + modelo baseline | concluída |
| 2 | Simulação e detecção de drift | concluída |
| 3 | Observabilidade de pipelines e modelos | pendente |
| 4 | Governança, LGPD e causalidade | pendente |
| 5 | Entrega final (README consolidado, CI, vídeo) | pendente |

## Dataset

**Default of Credit Card Clients** (UCI, 30.000 clientes, CC BY 4.0). Fonte, dicionário de dados e problemas
de qualidade encontrados estão em [`data/README.md`](data/README.md).

## Como executar

Requer **Python 3.12 ou superior** e `make`.

```bash
make setup
```

```bash
make all
```

`make all` executa o pipeline completo: verifica o dataset, prepara as camadas de dados, demonstra o
contrato, treina o modelo, simula 6 meses de produção e monitora o drift. Cada passo também pode ser
rodado isoladamente:

| Comando | O que faz |
|---|---|
| `make data` | Garante o dataset bruto e verifica sua integridade (SHA-256) |
| `make prepare` | Limpa, pseudonimiza e separa Referência x Pool de Produção |
| `make contract-demo` | Aprova um lote válido e bloqueia um lote corrompido |
| `make validate BATCH=arquivo.csv` | Valida um lote qualquer contra o contrato |
| `make train` | Treina os modelos e registra no MLflow |
| `make simulate` | Simula 6 meses de produção com drift crescente |
| `make drift` | Monitora os meses: contrato, predição, drift, desempenho e alertas |
| `make mlflow-ui` | Abre a interface do MLflow em http://127.0.0.1:5000 |
| `make test` / `make lint` | Testes automatizados e verificação de estilo |

## Etapa 1 — Validação de dados e contratos

### Preparação dos dados

O dado bruto passa por uma preparação documentada antes de qualquer uso
(`src/credit_monitoring/data/prepare.py`):

| Tratamento | Registros | Motivo |
|---|---|---|
| `education` 0, 5 e 6 → 4 ("outros") | 345 | Códigos não documentados ou "desconhecido" na fonte |
| `marriage` 0 → 3 ("outros") | 54 | Código não documentado na fonte |
| Remoção de registros idênticos | 35 | Não trazem informação nova e, divididos entre treino e teste, inflariam as métricas |
| Pseudonimização do `ID` | todos | HMAC-SHA256 com chave secreta mantida fora do repositório (LGPD, art. 13, §4º) |

Em seguida, o dataset é dividido **uma única vez**, de forma estratificada e com semente fixa:

| Camada | Registros | Inadimplência | Uso |
|---|---|---|---|
| Referência | 20.975 | 22,13% | Único dataset usado no treino |
| Pool de Produção | 8.990 | 22,12% | Nunca visto no treino; base dos lotes onde o drift vai acontecer |

### Contrato de dados (Pandera)

Todo lote passa por um **gate de ingestão** (`src/credit_monitoring/contracts/`). Se uma regra rígida
for violada, o lote **inteiro** é bloqueado e copiado para `data/quarantine/`, com relatório em HTML e
JSON. Pela linha de comando, o processo termina com código de saída 1.

| ID | Regra | Tipo |
|---|---|---|
| R1 | Schema estrito: colunas obrigatórias, sem colunas extras (evita vazamento de dados como CPF), tipos numéricos válidos | rígida |
| R2 | Sem valores nulos | rígida |
| R3 | Idade entre 18 e 100 anos | rígida |
| R4 | Limite de crédito maior que zero e até 5.000.000 | rígida |
| R5 | Domínios válidos: sexo, escolaridade, estado civil, status de pagamento (-2 a 9) e alvo | rígida |
| R6 | Valores pagos não negativos | rígida |
| R7 | Sem duplicatas: `client_id` único e nenhum registro reenviado | rígida |
| W1 | Fatura negativa (saldo credor) | alerta |
| W2 | Fatura mais recente acima de 1,5x o limite | alerta |

Os limites descrevem o que é **plausível para o negócio**, e não a distribuição do treino. Assim, um
lote com drift (ex.: limites maiores por inflação) continua válido: mudança de distribuição é papel do
monitoramento de drift (Etapa 2), não do contrato.

As checagens convertem o valor para número antes de comparar. Sem isso, um único texto inválido numa
coluna impediria o Pandera de checar os outros valores dela e esconderia outras violações.

**Demonstração** (`make contract-demo`): dois lotes de 500 registros chegam como CSV em `data/incoming/`.

| Lote | Resultado | Relatório |
|---|---|---|
| Válido | Aprovado, com alertas W1 (36 registros) e W2 (2) | [`reports/validation/lote_valido.html`](reports/validation/lote_valido.html) |
| Corrompido (erros injetados para cada regra) | **Bloqueado**: R1 a R7 detectadas, 19 registros com violação | [`reports/validation/lote_corrompido.html`](reports/validation/lote_corrompido.html) |

### Modelo baseline

Dois candidatos treinados na Referência (64% treino, 16% validação, 20% teste). O campeão e o limiar de
decisão são escolhidos **só na validação**; o teste serve apenas para reportar o desempenho final.

| Candidato | AUC validação | AUC teste | KS | Gini | F1 | Recall | Precisão |
|---|---|---|---|---|---|---|---|
| Regressão Logística | 0,745 | 0,726 | 0,386 | 0,451 | 0,515 | 0,490 | 0,542 |
| **XGBoost (campeão)** | **0,788** | **0,778** | **0,436** | **0,557** | **0,532** | **0,560** | **0,506** |

- **Limiar de decisão:** 0,275 (máximo F1 na validação), coerente com a taxa de inadimplência de ~22%.
- **Atributos fora do modelo:** `sex` e `marriage`, por minimização e não discriminação (LGPD, art. 6º,
  III e IX). Eles continuam no dataset apenas para a auditoria de vieses da Etapa 4.
- **Rastreamento:** os dois candidatos ficam registrados no MLflow; o campeão vai para o Model Registry
  como `credit-scoring-model` com o alias `champion`. Cada modelo é salvo no seu formato nativo e
  seguro, sem *pickle*: XGBoost em JSON/UBJ e scikit-learn via skops.
- **Metadados** do modelo (features, limiar, métricas, impressão digital dos dados e versões das
  bibliotecas) em [`models/model_metadata.json`](models/model_metadata.json).

## Etapa 2 — Simulação e detecção de drift

### Simulação da passagem do tempo (`make simulate`)

Seis lotes mensais de 1.400 clientes são amostrados do **Pool de Produção** (nunca visto no treino) e
transformados segundo um cenário econômico que piora mês a mês. Os cenários ficam declarados em
`config/settings.yaml` (seção `simulation`) e cada mecanismo é separado, para que a análise causal da
Etapa 4 consiga medir o efeito de cada um:

| Mês | Cenário | O que muda nos dados | Inadimplência real |
|---|---|---|---|
| M1 | Estável (controle) | Nada: prova que o monitoramento não gera falso alarme | 22,1% |
| M2 | Início da inflação | Faturas +10% | 22,6% |
| M3 | Inflação persistente | Faturas +20%; pagamentos não acompanham | 21,6% |
| M4 | Novos perfis | Faturas +25%, limites +20%, 45% de clientes até 28 anos; **um parceiro envia um lote corrompido** | 22,4% |
| M5 | Desemprego em alta | Tudo acima + 12% da carteira passa a atrasar e pagamentos caem 10% | 28,3% |
| M6 | Crise | Efeitos intensificados + **concept drift**: clientes em dia, porém muito endividados, passam a inadimplir sem mudar nenhuma variável | 39,4% |

Os rótulos seguem o mecanismo: inflação, limites e perfis mantêm o rótulo original (o risco real não
muda). No desemprego, o rótulo é sorteado com a taxa de inadimplência da Referência para o novo atraso,
preservando a relação que o modelo aprendeu. No concept drift, só o rótulo muda. Os lotes chegam sem o
alvo em `data/incoming/`; os rótulos, conhecidos só depois em produção, ficam em
`data/production/labels/`.

### Detecção (`make drift`)

Cada mês passa por: **gate de ingestão** (contrato da Etapa 1) → **predição** → **data drift** →
**prediction drift** → **desempenho** (com os rótulos) → **alertas**. A base de comparação é o conjunto de
teste da Referência: mesma distribuição do treino, mas fora do ajuste do modelo, então seus scores não
têm o viés otimista de dados de treino.

Cada variável recebe dois testes, que se complementam:

| Teste | Mede | Ferramenta | Limiar |
|---|---|---|---|
| **PSI** com faixas pelos decis da referência | Tamanho da mudança | implementação própria (`drift/stats.py`) | 0,10 moderado · 0,25 significativo |
| **Kolmogorov-Smirnov** | Se a mudança é estatisticamente significativa | Evidently AI (`DataDriftPreset`, método `ks`) | p-valor < 0,01 |

- **Por que PSI próprio:** o PSI do Evidently usa faixas de largura igual. Em variáveis de cauda longa,
  como faturas, quase todos os valores caem na primeira faixa e uma inflação de 30% gerava PSI de apenas
  0,055. Faixas por decis são o padrão em risco de crédito.
- **Por que α = 0,01 no KS:** com 23 testes por mês, α = 0,05 gerou algum falso positivo em 33% de 30
  meses simulados sem drift; com 0,01, em 3%.
- **Severidade:** *significativo* (PSI ≥ 0,25), *moderado* (PSI ≥ 0,10) ou *leve*, quando só o KS detecta
  a mudança. O nível leve funciona como detecção precoce, antes de o PSI subir.

### Resultados

| Mês | Variáveis com drift | Principais | PSI do score | AUC | Score médio | Inadimplência real | Alertas |
|---|---|---|---|---|---|---|---|
| M1 | 0/23 | — | 0,003 | 0,766 | 21,9% | 22,1% | 0 |
| M2 | 1/23 (leve) | `bill_amt_1` | 0,020 | 0,763 | 23,0% | 22,6% | 1 |
| M3 | 6/23 (leve) | `bill_amt_1..6` | 0,016 | 0,788 | 23,2% | 21,6% | 1 |
| M4 | 8/23 | `age` (moderado) | 0,007 | 0,801 | 22,7% | 22,4% | 4 |
| M5 | 20/23 | `pay_status_2` (significativo), `pay_status_1`, `pay_status_3`, `age` | 0,100 | 0,805 | 27,9% | 28,3% | 9 |
| M6 | 18/23 | `pay_status_2`, `pay_amt_1`, `pay_status_1` (significativos) | 0,132 | **0,725** | 29,5% | **39,4%** | 12 |

O que os resultados mostram:

1. **Sem falso alarme:** M1 não acusa drift em nenhuma variável.
2. **Detecção precoce:** a inflação de 10% em M2 já aparece no KS, mesmo com PSI baixo.
3. **Data drift nem sempre degrada o modelo:** em M4 e M5 a distribuição muda bastante, mas o AUC se
   mantém e o score médio acompanha a inadimplência real (27,9% x 28,3% em M5). O modelo ainda acerta.
4. **Concept drift degrada, e o score sozinho não mostra o tamanho do problema:** em M6 o AUC cai 0,053
   (alerta crítico) e o modelo **subestima o risco em 10 pontos** (29,5% previsto x 39,4% real).
5. **O contrato continua atuando em produção:** o lote corrompido do parceiro em M4 é bloqueado e vai
   para a quarentena, gerando alerta crítico de qualidade de dados.

### Relatórios gerados (`reports/drift/`)

| Arquivo | Conteúdo |
|---|---|
| [`drift_report.html`](reports/drift/drift_report.html) | **Relatório consolidado:** linha do tempo, mapa de PSI por variável e mês, AUC e calibração ao longo do tempo e alertas |
| `evidently_M1.html` … `evidently_M6.html` | Relatórios do Evidently por mês: drift por variável (KS) e desempenho de classificação |
| `feature_drift.csv` | PSI, p-valor do KS e severidade de cada variável em cada mês |
| `monthly_summary.csv` | Resumo mensal: drift, score, desempenho e alertas |
| `alerts.json` | Todos os alertas, com categoria, severidade, valor e limiar |

Os lotes mensais também geram relatório de validação do contrato em `reports/validation/` (incluindo
`M4_parceiro.html`, o lote bloqueado).

## Estrutura

```
├── config/settings.yaml        # configuração central (caminhos, sementes, limites, modelo)
├── data/                       # raw (versionado) e camadas geradas
├── models/                     # metadados do modelo (o binário é gerado por make train)
├── reports/
│   ├── validation/             # relatórios do contrato de dados (HTML + JSON)
│   └── drift/                  # relatórios de drift (Evidently, consolidado, CSVs e alertas)
├── src/credit_monitoring/
│   ├── data/                   # ingestão, limpeza, pseudonimização e separação
│   ├── contracts/              # contrato Pandera, gate de ingestão e relatórios
│   ├── models/                 # treino, métricas e artefato do modelo
│   ├── drift/                  # simulação de produção, PSI/KS, monitoramento e relatório
│   └── monitoring/             # integração com o MLflow
├── tests/                      # testes automatizados (pytest)
├── Makefile                    # atalhos de execução
├── requirements.txt            # dependências de execução (versões fixas)
└── requirements-dev.txt        # dependências de desenvolvimento
```
