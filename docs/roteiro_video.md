# Roteiro do vídeo STAR (até 5 minutos)

Roteiro para a gravação da apresentação do Tech Challenge Fase 4. Cada bloco traz o tempo, o que mostrar na
tela e um texto de narração. O texto tem cerca de 700 palavras (perto de 5 minutos em ritmo de fala normal);
adaptem livremente, mas mantenham os números, que são os que o pipeline gera.

## Antes de gravar

1. Rodar o projeto do zero para que tudo esteja atualizado:

   ```bash
   make all
   ```

2. Deixar abertas, nesta ordem, as abas do navegador:
   1. `reports/validation/lote_corrompido.html`
   2. `reports/drift/drift_report.html`
   3. `reports/drift/evidently_M6.html`
   4. `reports/monitoring/dashboard.html`
   5. MLflow (`make mlflow-ui` em outro terminal e abrir http://127.0.0.1:5000, experimento
      `credit-scoring-monitoring`, aba *Model metrics* da execução mais recente)
   6. `reports/governance/governance_report.html`
3. Deixar um terminal pronto com o comando de explicação. Pegue um `client_id` de um cliente recusado na tabela
   "Exemplos de motivos de recusa" do relatório de governança:

   ```bash
   make explain BATCH=data/incoming/M6.csv CLIENT=<client_id>
   ```

4. Zoom do navegador entre 110% e 125%, para os números ficarem legíveis no vídeo.

## Situation — o problema (0:00 a 0:35)

**Tela:** README no GitHub, no título e na seção "Resultados em um minuto".

> Uma fintech concede crédito com um modelo de *credit scoring* em produção. A economia mudou: inflação,
> desemprego e novos perfis de clientes. O risco é a **degradação silenciosa**: o modelo continua respondendo,
> sem nenhum erro técnico, mas passa a aprovar quem não vai pagar. Quando a inadimplência aparece no balanço,
> o prejuízo já aconteceu. Nosso trabalho foi construir a camada que percebe isso a tempo, explica a causa e
> garante que tudo esteja dentro da LGPD.

## Task — o que precisava ser feito (0:35 a 1:05)

**Tela:** README, seção "Atendimento aos requisitos".

> Os requisitos eram quatro: impedir que dados ruins cheguem ao modelo; simular a passagem do tempo e detectar
> o drift com testes estatísticos; dar visibilidade operacional ao pipeline, à infraestrutura e ao modelo; e
> documentar a governança, com privacidade, base legal, vieses e uma análise causal da degradação. Usamos o
> dataset público de inadimplência de cartão de crédito da UCI, com 30 mil clientes.

## Action — como fizemos (1:05 a 3:35)

### Contrato de dados (1:05 a 1:45)

**Tela:** `lote_corrompido.html`. Mostrar o selo BLOQUEADO e a regra R1 com a coluna `cpf`.

> Todo lote passa por um contrato de dados com o Pandera: sete regras que bloqueiam o lote, como idade entre
> 18 e 100 anos, limite positivo e ausência de duplicatas. Aqui um parceiro enviou um lote com erros de todos
> os tipos, inclusive uma coluna de CPF que não deveria existir. O lote inteiro foi bloqueado e foi para a
> quarentena: nenhum registro chegou ao modelo. Os limites do contrato descrevem o que é plausível para o
> negócio, e não a distribuição do treino, para que um drift legítimo não seja confundido com dado inválido.

### Simulação e detecção de drift (1:45 a 2:30)

**Tela:** `drift_report.html`. Mostrar a linha do tempo e o mapa de PSI; depois, rapidamente, `evidently_M6.html`.

> Separamos os dados uma única vez: uma Referência para treinar o XGBoost, com AUC de 0,778, e um pool de
> produção que o modelo nunca viu. A partir dele simulamos seis meses que pioram: inflação, mais clientes
> jovens, desemprego e, no último mês, uma crise. Cada variável é comparada com a referência por dois testes: o
> PSI, que mede o tamanho da mudança, e o teste de Kolmogorov-Smirnov do Evidently, que mede se ela é
> significativa. O mês de controle não gerou nenhum alarme falso, e uma inflação de só 10% já foi detectada no
> segundo mês.

### Observabilidade (2:30 a 3:05)

**Tela:** `dashboard.html`. Mostrar o semáforo e a grade de tarefas; depois a aba do MLflow.

> Cada etapa do pipeline grava um log estruturado, num arquivo central, com status, duração, memória e volume.
> O dashboard é montado a partir desse log: são 32 tarefas, 8.400 registros processados e 303 rejeitados. O
> semáforo mostra o pipeline em atenção, porque houve um lote bloqueado e um arquivo vazio, ambos tratados; a
> infraestrutura saudável; e o modelo em estado crítico no mês da crise. As mesmas métricas vão para o MLflow,
> mês a mês.

### Governança e LGPD (3:05 a 3:35)

**Tela:** terminal com `make explain`; depois o README, seção "Plano de proteção de dados".

> O identificador do cliente é pseudonimizado na entrada, sexo e estado civil ficam fora do modelo, e os logs
> não guardam dado pessoal. A base legal é a proteção do crédito, do artigo 7º da LGPD. Para o artigo 20, que
> garante a revisão de decisões automatizadas, qualquer decisão pode ser explicada: aqui, os três motivos que
> mais aumentaram o risco deste cliente recusado.

## Result — o que descobrimos (3:35 a 4:40)

**Tela:** `drift_report.html`, gráficos de AUC e de score x inadimplência; depois `governance_report.html`, a
análise causal e a tabela de prescrição.

> No quinto mês, vinte das vinte e três variáveis tinham drift, e mesmo assim o modelo acertava: o score médio
> acompanhava a inadimplência real. No sexto mês, a crise: o AUC caiu para 0,725 e o modelo previa 29,5% de
> inadimplência quando a real foi 39,4%. Os alertas críticos dispararam.
>
> Para saber a causa, ligamos e desligamos cada mecanismo da crise em todas as combinações, com 1.280
> simulações. O resultado: 87% da descalibração vêm do concept drift, clientes em dia, mas muito endividados,
> que passam a não pagar sem que nenhuma variável mude. O drift mais visível, o dos atrasos, até melhorou o
> AUC. E a solução foi testada: retreinar com os dados recentes recupera 0,038 de AUC.
>
> Na equidade, encontramos uma diferença acima do limite entre níveis de escolaridade, e mostramos que
> retirar idade e escolaridade do modelo reduz as disparidades sem perder desempenho.

## Lições aprendidas (4:40 a 5:00)

**Tela:** README, seção "Lições aprendidas".

> Três lições. Drift nos dados não é o mesmo que modelo errado. A causa mais grave pode não deixar rastro
> nas variáveis, e só os rótulos a revelam. E esconder um atributo sensível do modelo não basta para torná-lo
> justo. Obrigado!

## Checklist da entrega

- [ ] Vídeo com no máximo 5 minutos.
- [ ] Mostrar o relatório de drift apontando as variáveis degradadas (critério "Vídeo STAR").
- [ ] Mostrar os alertas gerados e explicar o impacto (AUC e risco subestimado).
- [ ] Publicar o vídeo (YouTube não listado ou Google Drive com acesso por link).
- [ ] Colocar o link no README, na linha "Vídeo de apresentação", no topo do arquivo.
