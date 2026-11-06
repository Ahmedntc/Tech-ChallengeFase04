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
| 1 | Validação de dados e contratos + modelo baseline | pendente |
| 2 | Simulação e detecção de drift | pendente |
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
make data
```

```bash
make test
```

`make help` lista todos os comandos disponíveis.

## Estrutura

```
├── config/settings.yaml        # configuração central (caminhos, sementes, dataset)
├── data/                       # raw (versionado) e camadas geradas
├── src/credit_monitoring/      # código do pipeline
├── tests/                      # testes automatizados (pytest)
├── Makefile                    # atalhos de execução
├── requirements.txt            # dependências de execução (versões fixas)
└── requirements-dev.txt        # dependências de desenvolvimento
```
