# Loteca ML — Estratégia 9 Secos / 5 Duplos / 0 Triplos

Projeto de Machine Learning para gerar **um único palpite final da Loteca**, com foco em maximizar a probabilidade de atingir **13 ou 14 acertos**, respeitando sempre:

- **9 secos**;
- **5 duplos**;
- **0 triplos**;
- **19 marcações** no total.

O projeto combina probabilidades por partida, ranking Top1/Top2/Top3, validação walk-forward, hard/soft constraints, backtesting e otimização do ticket.

> O objetivo principal não é maximizar accuracy jogo a jogo. A unidade final de avaliação é o **ticket completo de 19 marcações**, com prioridade para **P(>=13)**.

---

# Objetivo

Arquivos principais:

```text
data/concursos_anteriores.csv
data/proximo_concurso.csv
```

Para cada partida são usadas probabilidades normalizadas:

```text
p(1) = vitória do mandante
p(X) = empate
p(2) = vitória do visitante

p(1) + p(X) + p(2) = 1
```

Ranking:

```text
Top1 = resultado mais provável
Top2 = segundo resultado mais provável
Top3 = resultado menos provável
```

Desempate:

```text
1 > 2 > X
```

Base atual:

```text
445 concursos
30 concursos na janela histórica inicial
415 concursos testados em walk-forward

Top1: 51.7817%
Top2: 26.5329%
Top3: 21.6854%
```

---

# Função objetivo

A hierarquia de decisão é orientada à cauda superior:

```text
1. maior P13+
2. maior número de concursos com 14
3. maior número de concursos com 13
4. maior P12+
5. maior número de concursos com 12
6. maior média de acertos
7. menor instabilidade/variância
```

Accuracy, média, win rate da segunda marca, Brier Score, Log Loss e ECE são métricas importantes, mas funcionam principalmente como **diagnóstico**. Uma alteração só deve ser promovida quando melhorar o ticket fora da amostra.

---

# Princípio central — preservar o Top1

O `p(Top1)` continua sendo o baseline individual mais forte.

Uma métrica histórica só pode substituir ou reordenar o Top1 se demonstrar informação incremental fora da amostra.

Critérios mínimos:

```text
1. superar p(Top1) em walk-forward;
2. vencer quando sua ordenação discorda de p(Top1);
3. melhorar P13+ do ticket;
4. apresentar estabilidade temporal;
5. não usar informação futura.
```

Até o momento, nenhuma correção testada cumpriu esses critérios.

## Correções do Top1 — benchmarks congelados

```text
[DISAGREEMENT] top1_residual
3234 casos | baseline 802 x histórico 750 | neutros 1682
win rate histórico: 48.32%

[DISAGREEMENT] top1_lift
3264 casos | baseline 801 x histórico 755 | neutros 1708
win rate histórico: 48.52%

[DISAGREEMENT] top1_reliability
3323 casos | baseline 821 x histórico 746 | neutros 1756
win rate histórico: 47.61%

[TOP1-META]
Brier baseline: 0.233977
Brier meta:     0.240629

[DISAGREEMENT] p_top1_meta
4107 casos | baseline 1140 x meta 896 | neutros 2071
win rate meta: 44.01%
```

Conclusão:

> **`top1_residual`, `top1_lift`, `top1_reliability` e `p(top1_meta)` permanecem como benchmarks/telemetria e não alteram o ticket final.**

---

# Arquitetura atual

A arquitetura baseline divide o problema em duas decisões:

```text
14 jogos
   ↓
DoubleAllocator
   ↓
5 jogos recebem duplo
   ↓
SecondMarkSelector
   ↓
T1T2 ou T1T3
   ↓
Hard Constraints
   ↓
ticket final
```

## DoubleAllocator

Escolhe **quais cinco jogos recebem duplo**.

Políticas atuais:

```text
gain
top2_probability
uncertainty
margin
ratio
hist_top1
hist_top2
exact
```

## SecondMarkSelector

Escolhe **qual resultado acompanha Top1 em cada duplo**.

Candidatos:

```text
top2_baseline
recovery
threshold_recovery
second_mark_meta
double_value
```

---

# Estratégias atuais de alocação dos duplos

## gain / top2_probability

```text
score = p(Top2)
```

`gain` e `top2_probability` são equivalentes no baseline atual. `top2_probability` existe como nome explícito da regra; futuramente `gain` pode receber uma definição de valor marginal diferente.

## uncertainty

```text
score = 1 - p(Top1)
```

## margin

```text
score = 1 - (p(Top1) - p(Top2))
```

## ratio

```text
score = p(Top2) / p(Top1)
```

## hist_top1 / hist_top2

Benchmarks posicionais históricos.

## exact

Avalia:

```text
C(14,5) = 2.002
```

alocações dos cinco duplos e maximiza principalmente `P(>=13)` segundo as probabilidades disponíveis.

Importante distinguir:

```text
exact_probability = otimização ex-ante usando probabilidades
oracle_allocator  = diagnóstico ex-post usando resultados reais
```

---

# Estado atual do backtest

415 concursos fora da janela inicial:

```text
gain / top2_probability
14: 0 | 13: 6 | 12: 17 | P13+: 1.445783% | P12+: 5.542169%

uncertainty
14: 0 | 13: 6 | 12: 19 | P13+: 1.445783% | P12+: 6.024096%

margin
14: 0 | 13: 5 | 12: 18 | P13+: 1.204819% | P12+: 5.542169%

ratio
14: 0 | 13: 6 | 12: 17 | P13+: 1.445783% | P12+: 5.542169%

hist_top1
14: 0 | 13: 5 | 12: 17 | P13+: 1.204819% | P12+: 5.301205%

hist_top2
14: 0 | 13: 5 | 12: 20 | P13+: 1.204819% | P12+: 6.024096%

exact
14: 0 | 13: 6 | 12: 18 | P13+: 1.445783% | P12+: 5.783133%
```

Critério atual:

```text
P13+
↓
14
↓
13
↓
P12+
↓
estabilidade
↓
média
```

`uncertainty` permanece como política selecionada por desempate operacional. `gain`, `top2_probability`, `uncertainty`, `ratio` e `exact` ainda não apresentam separação robusta em P13+.

Telemetria recente:

```text
[ALLOCATOR OVERLAP]
uncertainty x gain:             4.299 / 5
uncertainty x top2_probability: 4.299 / 5
uncertainty x ratio:            4.728 / 5
uncertainty x exact:            4.658 / 5

[PAIRWISE] gain vs uncertainty
62 vitórias | 300 empates | 53 derrotas | delta médio +0.0217
```

O maior número de acertos totais do `gain` não implica superioridade em P13+.

---

# Error Recovery Score

```text
recovery_top2 = P(Top2_hit | Top1_miss, contexto)
recovery_top3 = P(Top3_hit | Top1_miss, contexto)
```

Baseline:

```text
Seco  = Top1
Duplo = Top1 + Top2
```

Resultado atual:

```text
[SECOND-MARK DISAGREEMENT]
739 casos
Top2 baseline wins: 368
recovery wins:      371
recovery win rate:  50.20%
seletor final:      top2_baseline
```

Thresholds:

```text
threshold   trocas   Top2   recovery   win rate   IC95%
0.00          739     368      371      50.20%    46.82%–53.86%
0.02          669     338      331      49.48%    45.74%–53.36%
0.05          549     262      287      52.28%    48.45%–56.47%
0.10          470     222      248      52.77%    48.30%–57.23%
0.15          342     161      181      52.92%    47.66%–58.19%
```

Todos os IC95% ainda incluem 50%.

## Nested Recovery

```text
                         Top2 baseline    Nested recovery
14                              0                 0
13                              6                 5
12                             19                31
P13+                       1.4458%           1.2048%
P12+                       6.0241%           8.6747%
média                       8.7205            8.7759

delta P13+: -0.2410 p.p.
delta P12+: +2.6506 p.p.
```

Conclusão:

> O recovery melhora P12+ e média, mas reduz P13+. Portanto, **não é promovido** e `top2_baseline` permanece ativo.

---

# Prioridade 1 — Oracle Decomposition

Antes de aumentar a complexidade do ML, medir o teto de melhoria disponível.

## OracleAllocator

Mantém:

```text
Seco  = Top1
Duplo = Top1 + Top2
```

mas encontra retrospectivamente os cinco jogos que deveriam ter recebido duplo.

## OracleSecondMark

Mantém os cinco duplos da política real e escolhe retrospectivamente Top2 ou Top3 em cada um.

## OracleFull

Escolhe retrospectivamente:

```text
quais 5 jogos recebem duplo
+
Top2 ou Top3 em cada duplo
```

Os oráculos nunca participam da previsão final. Eles medem apenas o **teto estrutural** de melhoria.

Telemetria:

```text
[ORACLE DECOMPOSITION]
baseline_P13+:
allocator_oracle_P13+:
selector_oracle_P13+:
full_oracle_P13+:

baseline_P12+:
allocator_oracle_P12+:
selector_oracle_P12+:
full_oracle_P12+:
```

---

# Regret por componente

## Allocator regret

```text
allocator_regret = hits_oracle_allocator - hits_policy
```

## Distribution regret

```text
distribution_regret = hits_oracle_distribution - hits_selected_distribution
```

## Selector regret

```text
selector_regret = hits_oracle_selector - hits_selected_selector
```

## Full regret

```text
full_regret = hits_oracle_full - hits_ticket
```

Registrar:

```text
mean_regret
median_regret
regret_0_rate
regret_1_rate
regret_2plus_rate
max_regret
```

Isso permite descobrir em qual componente existe maior espaço real de melhoria.

---

# Prioridade 2 — DistributionBacktest

A próxima expansão importante é **avaliar todas as distribuições viáveis das 19 marcações entre Top1, Top2 e Top3**, antes de assumir que `14 Top1 + 5 Top2` é estruturalmente ótimo.

## Fase segura — Top1 sempre coberto

Enquanto não existir evidência forte para remover Top1, os 14 jogos continuam contendo Top1. As cinco marcações extras podem ser distribuídas entre Top2 e Top3:

```text
T1 / T2 / T3
14 / 5 / 0   ← baseline atual
14 / 4 / 1
14 / 3 / 2
14 / 2 / 3
14 / 1 / 4
14 / 0 / 5
```

Essas seis composições preservam:

```text
14 marcações Top1
5 marcações adicionais
19 marcações totais
9 secos
5 duplos
0 triplos
```

## Ponto essencial — distribuição não é posicionamento

Saber que `14/4/1` é uma boa composição não responde:

```text
quais 5 jogos devem receber duplo?
quais 4 recebem Top2?
qual recebe Top3?
```

Portanto, cada distribuição deve ser avaliada junto da escolha das posições.

Arquitetura proposta:

```text
DistributionSelector
        ↓
DoubleAllocator
        ↓
SecondMarkSelector / MarkingSelector
        ↓
Hard Constraints
        ↓
Ticket
```

---

# DistributionBacktest — métricas

Para cada distribuição registrar:

```text
14
13
12
11
10
<=9
P13+
P12+
mean
median
stddev
```

Telemetria esperada:

```text
[DISTRIBUTION BACKTEST]

T1/T2/T3     14   13   12    P13+     P12+     mean
14/5/0        ...  ...  ...     ...       ...      ...
14/4/1        ...  ...  ...     ...       ...      ...
14/3/2        ...  ...  ...     ...       ...      ...
14/2/3        ...  ...  ...     ...       ...      ...
14/1/4        ...  ...  ...     ...       ...      ...
14/0/5        ...  ...  ...     ...       ...      ...
```

A seleção deve respeitar a função objetivo do projeto, com P13+ como métrica principal.

---

# OracleDistribution

Criar um diagnóstico retrospectivo adicional:

```text
[ORACLE DISTRIBUTION]
```

Para cada concurso, usando o resultado real somente para diagnóstico:

> qual das distribuições permitidas teria produzido o maior número de acertos após otimizar o posicionamento das cinco marcações extras?

Registrar frequência de ótimo:

```text
14/5/0: ... concursos
14/4/1: ... concursos
14/3/2: ... concursos
14/2/3: ... concursos
14/1/4: ... concursos
14/0/5: ... concursos
```

O OracleDistribution mede o teto potencial de abandonar a composição fixa `14/5/0`.

---

# Nested Distribution Selection

Não selecionar a melhor distribuição usando os mesmos 415 concursos em que ela é avaliada.

Fluxo obrigatório:

```text
histórico disponível até N
        ↓
avaliar distribuições somente nesse passado
        ↓
selecionar distribuição
        ↓
congelar a escolha
        ↓
aplicar no concurso N+1
        ↓
registrar resultado
        ↓
incluir N+1 no histórico
        ↓
repetir
```

Exemplo:

```text
Concursos 1..100 → seleciona distribuição
Concurso 101     → teste real
Concursos 1..101 → seleciona novamente
Concurso 102     → teste real
```

Somente o resultado **nested walk-forward** pode promover uma distribuição diferente do baseline.

Telemetria:

```text
[NESTED DISTRIBUTION]
selected_usage:
14/5/0: ...
14/4/1: ...
14/3/2: ...
...

baseline P13+:
nested P13+:
delta P13+:

baseline P12+:
nested P12+:
delta P12+:
```

---

# Matriz Distribution × Allocator

A melhor distribuição pode depender de como os cinco duplos são posicionados.

Comparar:

```text
                    14/5/0  14/4/1  14/3/2  14/2/3  14/1/4  14/0/5

top2_probability       ...      ...      ...      ...      ...      ...
uncertainty             ...      ...      ...      ...      ...      ...
margin                  ...      ...      ...      ...      ...      ...
ratio                   ...      ...      ...      ...      ...      ...
exact_probability       ...      ...      ...      ...      ...      ...
```

Cada célula deve registrar no mínimo:

```text
P13+
P12+
n14
n13
n12
mean
stddev
```

---

# Matriz Distribution × Allocator × Selector

Depois do teste bidimensional, expandir para:

```text
Distribution
     ×
DoubleAllocator
     ×
SecondMarkSelector
```

O objetivo é separar três perguntas:

```text
1. quantos Top2/Top3 devem existir nas cinco marcações extras?
2. quais cinco jogos devem receber duplo?
3. quais desses duplos devem usar Top2 ou Top3?
```

A unidade final de comparação continua sendo o **ticket completo**.

---

# DistributionSelector condicionado ao concurso

Somente depois de validar que mais de uma distribuição apresenta sinal fora da amostra, estudar seleção dinâmica por perfil do concurso.

Features candidatas:

```text
mean_p_top1
mean_p_top2
mean_p_top3
mean_gap_12
mean_gap_23
min_gap_12
n_equilibrated_games
mean_entropy
max_entropy
```

Exemplo conceitual:

```text
concurso muito concentrado → 14/5/0
concurso com T2/T3 próximos → 14/4/1 ou 14/3/2
```

Qualquer regra, árvore, meta-modelo ou threshold deve ser treinado e selecionado dentro de nested walk-forward.

---

# Expansão posterior — secos Top2/Top3

Somente depois da fase segura, permitir distribuições com menos de 14 marcações Top1.

Exemplos:

```text
13/5/1
12/6/1
12/5/2
11/6/2
...
```

Isso implica que algum jogo pode ter:

```text
SECO Top2
ou
SECO Top3
```

Essa expansão exige evidência muito mais forte porque as correções atuais do Top1 não superaram o baseline.

Regra de promoção:

> distribuições com menos de 14 Top1 só podem entrar no ticket se superarem o baseline em nested walk-forward, P13+ e testes de robustez.

---

# FullMarkingOptimizer

Última etapa da expansão do espaço de busca.

Permitir por jogo:

```text
SECO:
T1
T2
T3

DUPLO:
T1T2
T1T3
T2T3
```

Sempre respeitando:

```text
9 secos
5 duplos
0 triplos
19 marcações
```

O espaço completo é grande. Portanto, evitar brute force ingênuo e usar busca hierárquica:

```text
DistributionSelector
        ↓
DoubleAllocator
        ↓
MarkingSelector
        ↓
ConstraintEngine
        ↓
TicketScorer
```

O FullMarkingOptimizer só deve ser implementado depois de o DistributionBacktest demonstrar que há ganho robusto ao expandir além de `14/5/0`.

---

# Comparação pareada

Para cada par de estratégias registrar:

```text
A > B
A = B
A < B
mean_delta_hits
```

Também registrar distribuição de deltas:

```text
+3
+2
+1
 0
-1
-2
-3
```

Para P13+:

```text
A >=13 e B <13
B >=13 e A <13
ambos >=13
nenhum >=13
```

Repetir para P12+.

---

# Bootstrap e significância

Usar comparação pareada por concurso:

```text
bootstrap >= 1.000 reamostragens
IC95% de ΔP13+
IC95% de ΔP12+
IC95% de Δmean
probabilidade empírica de A > B
```

Devido ao baixo número de concursos com 13+, diferenças de um ou dois concursos não devem ser interpretadas como evidência forte.

---

# Features de valor marginal

## gap_12

```text
gap_12 = p(Top1) - p(Top2)
```

## gap_23

```text
gap_23 = p(Top2) - p(Top3)
```

## entropia

```text
H = -Σ p(i) × log(p(i))
```

## double_value

```text
double_value ≈ P(Top1_miss) × P(second_mark_hit | Top1_miss, contexto)
```

Essas features devem primeiro servir como diagnóstico e somente depois alimentar seletores treinados.

---

# Estabilidade temporal

Comparar:

```text
primeiro terço
segundo terço
último terço
```

Posteriormente:

```text
expanding window
rolling 50
rolling 100
rolling 200
```

Decay temporal candidato:

```text
half-life 25
half-life 50
half-life 100
half-life 200
sem decay
```

A escolha da janela ou decay deve ocorrer dentro do nested walk-forward.

---

# Calibração

Diagnóstico atual:

```text
Brier multiclasse: 0.588408
Log Loss:          0.985557
ECE:               0.012378
```

Adicionar reliability tables separadas para Top1/Top2/Top3 e salvar:

```text
output/calibration_top1.csv
output/calibration_top2.csv
output/calibration_top3.csv
```

---

# Hard Constraints

Todo ticket final deve conter exatamente:

```text
9 secos
5 duplos
0 triplos
19 marcações
```

## Flamengo

Quando o **FLAMENGO/RJ** participar, sua vitória deve obrigatoriamente estar coberta:

```text
Flamengo mandante  → incluir 1
Flamengo visitante → incluir 2
```

## Palmeiras — Soft Constraint

Favorecer, quando o custo probabilístico for pequeno, soluções que excluam a vitória do **PALMEIRAS/SP**.

Limiar atual:

```text
0.03
```

O limiar é experimental e só deve ser alterado mediante validação adequada.

---

# Testes automatizados obrigatórios

Garantir permanentemente:

```text
14 jogos por concurso
9 secos
5 duplos
0 triplos
19 marcações
vitória do Flamengo coberta
probabilidades somando 1
Top1/Top2/Top3 distintos
desempate 1 > 2 > X
nenhum vazamento temporal
```

No baseline seguro:

```text
Top1 coberto nos 14 jogos
```

Durante DistributionBacktest:

```text
count(T1) + count(T2) + count(T3) = 19 marcações de ranking
exatamente 5 jogos com duas marcações
nenhum jogo com três marcações
```

Em toda rotina histórica:

```python
assert train_contest < test_contest
```

---

# Controle de experimentos

Criar/manter:

```text
output/experiments.csv
```

Campos sugeridos:

```text
timestamp
model
distribution
allocator
second_mark_selector
threshold
window
decay
features
n14
n13
n12
P13+
P12+
mean
stddev
git_commit
```

---

# Telemetria resumida

Ao final de cada execução:

```text
[SUMMARY]
Top1 accuracy:
Selected distribution:
Selected allocator:
Selected second mark:
Historical P13+:
Historical P12+:
Best experimental P13+:

Oracle allocator P13+:
Oracle distribution P13+:
Oracle selector P13+:
Oracle full P13+:

Current contest P14:
Current contest P13:
Current contest P13+:
Current contest E[hits]:
```

---

# Estrutura do repositório

```text
loteca-ML-9s-5d-0t/
├── main.py
├── data/
│   ├── concursos_anteriores.csv
│   └── proximo_concurso.csv
├── scripts/
│   ├── common.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── predict_results.py
├── models/
│   └── model.json
├── output/
│   ├── predictions.csv
│   ├── backtest.csv
│   └── experiments.csv        # planejado
├── tests/
└── README.md
```

---

# Execução

```powershell
python main.py
```

Testes:

```powershell
python -m unittest discover -v
```

---

# Roadmap — ordem prática

## Concluído

- [x] pipeline único de constraints;
- [x] invariantes 9/5/0 e 19 marcações;
- [x] políticas `gain`, `top2_probability`, `uncertainty`, `margin`, `ratio`, `hist_top1`, `hist_top2` e `exact`;
- [x] `P(14)`, `P(13)`, `P(>=13)` e `E[acertos]`;
- [x] walk-forward sem vazamento temporal;
- [x] backtest 10–14;
- [x] P13+ e P12+ empíricos;
- [x] Disagreement Test do Top1;
- [x] `top1_residual`, `top1_lift`, `top1_reliability`;
- [x] `p(top1_meta)`;
- [x] evidência para congelar correções do Top1;
- [x] Error Recovery Score;
- [x] Second-Mark Disagreement;
- [x] thresholds de recovery;
- [x] IC95% por threshold;
- [x] nested walk-forward do recovery;
- [x] evidência para manter `top2_baseline`;
- [x] overlap entre allocators;
- [x] comparação pareada inicial `gain vs uncertainty`.

## Fase 1 — diagnóstico estrutural

1. [x] implementar `OracleAllocator`;
2. [x] implementar `OracleSecondMark`;
3. [x] implementar `OracleFull`;
4. [x] implementar regret por allocator/selector/full;
5. [ ] adicionar pairwise específico de P13+/P12+;
6. [ ] adicionar distribuição dos deltas de acertos.

## Fase 2 — distribuição das 19 marcações

7. [ ] implementar `DistributionBacktest` seguro `14/5/0 → 14/0/5`;
8. [ ] otimizar posicionamento dentro de cada distribuição;
9. [ ] implementar `OracleDistribution`;
10. [ ] implementar regret de distribuição;
11. [ ] implementar `NestedDistributionSelector`;
12. [ ] registrar frequência de uso de cada distribuição no nested;
13. [ ] comparar `Distribution × Allocator`;
14. [ ] comparar `Distribution × Allocator × Selector`;
15. [ ] bootstrap pareado das melhores distribuições.

## Fase 3 — seleção dinâmica e valor marginal

16. [ ] adicionar `gap_12`;
17. [ ] aprofundar `gap_23`;
18. [ ] adicionar entropia;
19. [ ] implementar `double_value_score`;
20. [ ] testar `DistributionSelector` condicionado ao perfil do concurso;
21. [ ] implementar `second_mark_meta` somente se houver sinal suficiente.

## Fase 4 — robustez temporal e calibração

22. [ ] comparar expanding × rolling windows;
23. [ ] testar decay temporal dentro do nested;
24. [ ] gerar reliability tables de Top1/Top2/Top3;
25. [ ] salvar CSVs de calibração;
26. [ ] criar/manter `output/experiments.csv` com commit do Git.

## Fase 5 — FullMarkingOptimizer

27. [ ] permitir distribuições com menos de 14 Top1 somente após evidência nested;
28. [ ] avaliar secos Top2/Top3;
29. [ ] avaliar duplo `T2T3`;
30. [ ] implementar `FullMarkingOptimizer` hierárquico;
31. [ ] comparar FullMarkingOptimizer com baseline em P13+;
32. [ ] remover/substituir desempates arbitrários do `exact`;
33. [ ] otimizar o limiar do Palmeiras usando validação adequada.

---

# Critério de promoção

Uma estratégia experimental só pode substituir o baseline quando:

```text
melhorar P13+ fora da amostra
↓
não depender de seleção retrospectiva de hiperparâmetros
↓
apresentar resultado pareado favorável
↓
apresentar IC/bootstrap compatível com ganho real
↓
manter estabilidade temporal
↓
respeitar todas as Hard Constraints
```

Melhorar apenas P12+, média, accuracy ou win rate individual **não é suficiente** quando P13+ piora.

---

# Princípio geral

```text
p(Top1) preservado
      +
Oracle Decomposition
      +
DistributionBacktest
      +
Nested Distribution Selection
      +
DoubleAllocator
      +
SecondMarkSelector
      +
Regret / comparação pareada
      +
Incerteza estatística
      +
Hard Constraints
      +
Otimização de P13+
      ↓
PALPITE FINAL
```

> **Não promover a melhor regra observada; promover apenas a regra que continuar melhor quando escolhida usando somente o passado e avaliada prospectivamente.**
