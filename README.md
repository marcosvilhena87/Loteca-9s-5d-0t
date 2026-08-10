# Loteca ML — Estratégia 9 Secos / 5 Duplos / 0 Triplos

Projeto de Machine Learning para gerar **um único palpite final da Loteca**, com foco em maximizar a probabilidade de atingir **13 ou 14 acertos**, respeitando sempre:

- **9 secos**;
- **5 duplos**;
- **0 triplos**;
- **19 marcações** no total.

O projeto combina probabilidades por partida, ranking Top1/Top2/Top3, walk-forward, hard/soft constraints, backtesting, oráculos diagnósticos e otimização do ticket.

> O objetivo principal não é maximizar accuracy jogo a jogo. A unidade final é o **ticket completo de 19 marcações**, com prioridade para **P(>=13)**.

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

Accuracy, média, win rate da segunda marca, Brier Score, Log Loss e ECE são métricas diagnósticas. Uma alteração só deve ser promovida quando melhorar o **ticket fora da amostra**.

---

# Princípio central — preservar o Top1

O `p(Top1)` continua sendo o baseline individual mais forte.

Uma métrica histórica só pode substituir ou reordenar Top1 se demonstrar informação incremental fora da amostra.

Critérios mínimos:

```text
1. superar p(Top1) em walk-forward;
2. vencer quando sua ordenação discorda de p(Top1);
3. melhorar P13+ do ticket;
4. apresentar estabilidade temporal;
5. não usar informação futura.
```

Resultados atuais das correções:

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

# Arquitetura baseline

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

Escolhe **qual resultado acompanha Top1** em cada duplo.

Candidatos:

```text
top2_baseline
recovery
threshold_recovery
second_mark_meta
double_value
```

---

# Estratégias atuais de alocação

## gain / top2_probability

```text
score = p(Top2)
```

Atualmente são equivalentes. `top2_probability` funciona como nome explícito do baseline; `gain` pode futuramente receber uma definição aprendida de valor marginal.

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

Distinguir:

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

`uncertainty` permanece selecionada por desempate operacional. Não existe ainda separação robusta em P13+ entre `gain`, `top2_probability`, `uncertainty`, `ratio` e `exact`.

Telemetria:

```text
[ALLOCATOR OVERLAP]
uncertainty x gain:             4.299 / 5
uncertainty x top2_probability: 4.299 / 5
uncertainty x ratio:            4.728 / 5
uncertainty x exact:            4.658 / 5

[PAIRWISE] gain vs uncertainty
62 vitórias | 300 empates | 53 derrotas | delta médio +0.0217
```

O maior número total de acertos do `gain` não implica superioridade em P13+.

---

# Error Recovery Score

```text
recovery_top2 = P(Top2_hit | Top1_miss, contexto)
recovery_top3 = P(Top3_hit | Top1_miss, contexto)
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

> O recovery simples melhora P12+ e média, mas reduz P13+. Portanto, **não é promovido** e `top2_baseline` permanece ativo.

O resultado dos oráculos, porém, mostra que o **problema de segunda marca continua importante**; o sinal existe no teto estrutural, mas o modelo de recovery atual ainda não consegue capturá-lo de forma prospectiva.

---

# Oracle Decomposition — implementado

Os oráculos usam o resultado real **somente para diagnóstico retrospectivo**. Nunca entram na previsão final.

## OracleAllocator

Mantém:

```text
Seco  = Top1
Duplo = Top1 + Top2
```

mas encontra os cinco jogos que deveriam retrospectivamente ter recebido duplo.

## OracleSecondMark

Mantém os cinco duplos da política real e escolhe retrospectivamente Top2 ou Top3 em cada um.

## OracleFull

Escolhe retrospectivamente:

```text
quais 5 jogos recebem duplo
+
Top2 ou Top3 em cada duplo
```

## Resultado atual

```text
[ORACLE DECOMPOSITION]

baseline
P13+:  1.45%
P12+:  6.02%
média: 8.7205

allocator oracle
P13+:  11.08%
P12+:  31.08%
média: 10.7229

selector oracle
P13+:  5.54%
P12+:  21.45%
média: 10.1831

full oracle
P13+:  41.93%
P12+:  65.06%
média: 12.0289
```

Leitura:

- o `OracleAllocator` mostra um teto muito alto para melhorar **onde colocar os cinco duplos**;
- o `OracleSecondMark` mostra que há ganho relevante em escolher corretamente `T1T2` versus `T1T3`;
- o `OracleFull` mostra forte **interação entre allocator e selector**;
- a arquitetura atual ainda está longe de seu teto retrospectivo.

> O resultado não significa que P13+ de 41.93% seja alcançável ex-ante. Ele quantifica apenas o potencial estrutural disponível quando a decisão perfeita é conhecida retrospectivamente.

---

# Regret por componente — implementado

```text
[REGRET ALLOCATOR]
média:  2.0024
zero:   8.92%
2+:     67.23%
máximo: 5

[REGRET SELECTOR]
média:  1.4627
zero:   18.07%
2+:     44.34%
máximo: 4

[REGRET FULL]
média:  3.3084
zero:   0.96%
2+:     95.42%
máximo: 5
```

Definições:

```text
allocator_regret = hits_oracle_allocator - hits_policy
selector_regret  = hits_oracle_selector  - hits_selected_selector
full_regret      = hits_oracle_full      - hits_ticket
```

Interpretação principal:

> o `DoubleAllocator` é hoje um gargalo muito relevante: em média, aproximadamente **2 acertos por concurso** separam a política atual do melhor posicionamento retrospectivo dos cinco duplos.

---

# Nova métrica — Oracle Capture Rate

Regret mede perda absoluta. Uma métrica complementar deve medir **quanto do ganho disponível pelas cinco marcações extras é capturado**.

Para cada concurso:

```text
capture_rate =
    (hits_policy - hits_top1_only)
    /
    (hits_oracle_full - hits_top1_only)
```

Quando o denominador for zero, registrar o concurso separadamente como `no_oracle_gain_available`.

Telemetria:

```text
[ORACLE CAPTURE]
policy             mean_capture   median_capture
uncertainty        ...            ...
top2_probability   ...            ...
ratio              ...            ...
exact_probability  ...            ...
```

Objetivo:

> medir qual fração do valor potencial das cinco marcações extras a política consegue capturar.

---

# Prioridade 1 — OracleDistribution + DistributionBacktest

A próxima expansão é avaliar todas as distribuições seguras das 19 marcações entre Top1, Top2 e Top3, sem remover Top1 inicialmente.

## Fase segura — Top1 sempre coberto

```text
T1 / T2 / T3
14 / 5 / 0   ← baseline atual
14 / 4 / 1
14 / 3 / 2
14 / 2 / 3
14 / 1 / 4
14 / 0 / 5
```

Todas preservam:

```text
14 marcações Top1
5 marcações adicionais
19 marcações totais
9 secos
5 duplos
0 triplos
```

## Distribuição não é posicionamento

Saber que `14/4/1` é uma composição candidata não responde:

```text
quais 5 jogos recebem duplo?
quais 4 recebem Top2?
qual recebe Top3?
```

Cada distribuição deve ser avaliada junto do posicionamento das marcações extras.

---

# OracleDistribution

Para cada concurso, usando o resultado real apenas para diagnóstico:

> qual distribuição segura teria produzido mais acertos depois de otimizar também o posicionamento das cinco marcações extras?

Telemetria:

```text
[ORACLE DISTRIBUTION]
14/5/0: ... concursos
14/4/1: ... concursos
14/3/2: ... concursos
14/2/3: ... concursos
14/1/4: ... concursos
14/0/5: ... concursos

oracle_distribution_P13+: ...
oracle_distribution_P12+: ...
```

Também calcular:

```text
distribution_regret = hits_oracle_distribution - hits_selected_distribution
```

O OracleDistribution responde se existe potencial estrutural real em abandonar a composição fixa `14/5/0`.

---

# DistributionBacktest

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

Telemetria:

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

A função objetivo permanece centrada em P13+.

---

# Nested Distribution Selector

Não selecionar a distribuição vencedora usando o mesmo período em que ela é avaliada.

```text
histórico até N
      ↓
comparar distribuições no passado
      ↓
selecionar distribuição
      ↓
congelar
      ↓
aplicar no concurso N+1
      ↓
registrar
      ↓
repetir
```

Somente o resultado nested pode promover uma distribuição diferente de `14/5/0`.

Telemetria:

```text
[NESTED DISTRIBUTION]
usage 14/5/0: ...
usage 14/4/1: ...
usage 14/3/2: ...
...

baseline P13+: ...
nested P13+:   ...
delta P13+:    ...

baseline P12+: ...
nested P12+:   ...
delta P12+:    ...
```

---

# Prioridade 2 — Opportunity Dataset

Os oráculos devem servir também para descobrir **quais sinais pré-jogo explicam o valor das cinco marcações extras**.

Criar dataset diagnóstico por partida:

```text
contest
game
p_top1
p_top2
p_top3
gap_12
gap_23
entropy
top1_result
top2_result
top3_result

top1_hit
top2_hit
top3_hit
recoverable_by_top2
recoverable_by_top3
```

Targets locais recomendados:

```text
top1_miss
recoverable_by_top2
recoverable_by_top3
```

Não usar diretamente `oracle_selected_double` como target principal, porque essa seleção depende da limitação global de cinco vagas e dos outros 13 jogos do concurso.

Separação desejada:

```text
modelo local de valor por jogo
        +
otimização global das 5 vagas
```

---

# Prioridade 3 — DoubleValueModel

Em vez de selecionar duplos apenas por `p(Top2)` ou `1-p(Top1)`, aprender o **valor esperado da marcação adicional**.

Features candidatas:

```text
p_top1
p_top2
p_top3
gap_12
gap_23
ratio_top2_top1
ratio_top3_top2
entropy
identidade Top1/Top2/Top3
perfil probabilístico do concurso
```

Targets locais:

```text
value_T1T2 = P(Top2_hit | contexto)
value_T1T3 = P(Top3_hit | contexto)
```

ou, explicitamente condicionado ao erro do Top1:

```text
value_T1T2 ≈ P(Top1_miss) × P(Top2_hit | Top1_miss, contexto)
value_T1T3 ≈ P(Top1_miss) × P(Top3_hit | Top1_miss, contexto)
```

Todo treinamento e seleção de hiperparâmetros deve respeitar walk-forward/nested walk-forward.

---

# Prioridade 4 — JointDoubleOptimizer

Os resultados dos oráculos mostram que `DoubleAllocator` e `SecondMarkSelector` interagem fortemente.

Além da arquitetura sequencial:

```text
DoubleAllocator
      ↓
SecondMarkSelector
```

implementar uma arquitetura conjunta:

```text
para cada jogo:
    value_T1T2
    value_T1T3
        ↓
melhor alternativa de proteção por jogo
        ↓
otimização global das 5 vagas
        ↓
5 pares (jogo, tipo_de_duplo)
```

A unidade de decisão passa de:

```text
"este jogo merece duplo?"
```

para:

```text
"quanto vale T1T2 neste jogo?"
"quanto vale T1T3 neste jogo?"
```

O objetivo é capturar parte da grande diferença observada entre `OracleAllocator`, `OracleSecondMark` e `OracleFull` sem usar informação futura.

---

# Matriz Distribution × Optimizer

Depois de implementar DistributionBacktest e JointDoubleOptimizer, comparar:

```text
                      14/5/0  14/4/1  14/3/2  14/2/3  14/1/4  14/0/5

top2_probability         ...      ...      ...      ...      ...      ...
uncertainty               ...      ...      ...      ...      ...      ...
exact_probability         ...      ...      ...      ...      ...      ...
double_value_model        ...      ...      ...      ...      ...      ...
joint_double_optimizer    ...      ...      ...      ...      ...      ...
```

Cada célula deve registrar:

```text
n14
n13
n12
P13+
P12+
mean
stddev
```

---

# Comparação pareada

Para cada par de estratégias registrar:

```text
A > B
A = B
A < B
mean_delta_hits
```

Também registrar distribuição dos deltas:

```text
+5
+4
+3
+2
+1
 0
-1
-2
-3
-4
-5
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

Devido ao pequeno número atual de concursos com 13+, diferenças de um ou dois concursos não devem ser interpretadas como evidência forte.

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

Todo ticket deve conter exatamente:

```text
14 jogos
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

# FullMarkingOptimizer — etapa posterior

Somente depois de existir evidência nested para expandir além da fase segura, permitir:

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

Sempre respeitando 9/5/0 e 19 marcações.

Antes disso, o projeto deve tentar capturar uma parcela maior do enorme teto já identificado **sem remover Top1 dos 14 jogos**.

Arquitetura hierárquica futura:

```text
DistributionSelector
        ↓
ValueModel
        ↓
JointMarkingOptimizer
        ↓
ConstraintEngine
        ↓
TicketScorer
```

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
count(T1) + count(T2) + count(T3) = 19
exatamente 5 jogos com duas marcações
nenhum jogo com três marcações
```

Em toda rotina histórica:

```python
assert train_contest < test_contest
```

Oráculos:

```text
nunca alimentar previsão final
nunca alimentar diretamente features pré-jogo
servir apenas para diagnóstico, labels locais controlados e teto estrutural
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
optimizer
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
oracle_capture_rate
git_commit
```

---

# Telemetria resumida

Ao final de cada execução:

```text
[SUMMARY]
Top1 accuracy:
Selected distribution:
Selected allocator/optimizer:
Selected second mark:
Historical P13+:
Historical P12+:
Best experimental P13+:

Oracle allocator P13+: 11.08%
Oracle selector P13+:   5.54%
Oracle full P13+:      41.93%
Oracle distribution P13+: ...

Allocator regret mean: 2.0024
Selector regret mean:  1.4627
Full regret mean:      3.3084
Oracle capture rate:   ...

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
│   ├── experiments.csv          # planejado
│   └── opportunity_dataset.csv  # planejado
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
- [x] thresholds + IC95%;
- [x] nested walk-forward do recovery;
- [x] evidência para manter `top2_baseline`;
- [x] overlap entre allocators;
- [x] comparação pareada inicial;
- [x] `OracleAllocator`;
- [x] `OracleSecondMark`;
- [x] `OracleFull`;
- [x] regret allocator/selector/full;
- [x] quantificação do teto estrutural da arquitetura atual.

## Fase 1 — distribuição das cinco marcações extras

1. [ ] implementar `OracleDistribution`;
2. [ ] implementar `DistributionBacktest` seguro `14/5/0 → 14/0/5`;
3. [ ] otimizar posicionamento dentro de cada distribuição;
4. [ ] implementar `distribution_regret`;
5. [ ] implementar `NestedDistributionSelector`;
6. [ ] registrar frequência de uso de cada distribuição no nested;
7. [ ] bootstrap pareado das melhores distribuições.

## Fase 2 — aprender valor marginal

8. [ ] criar `output/opportunity_dataset.csv`;
9. [ ] adicionar `gap_12`, `gap_23` e entropia;
10. [ ] criar targets locais `top1_miss`, `recoverable_by_top2`, `recoverable_by_top3`;
11. [ ] implementar `DoubleValueModel`;
12. [ ] comparar `p(Top2)` vs valor aprendido;
13. [ ] validar tudo em walk-forward/nested.

## Fase 3 — otimização conjunta

14. [ ] implementar `JointDoubleOptimizer`;
15. [ ] estimar `value_T1T2` e `value_T1T3` por jogo;
16. [ ] otimizar conjuntamente jogo + tipo de duplo;
17. [ ] implementar `Oracle Capture Rate`;
18. [ ] comparar `Distribution × JointOptimizer`;
19. [ ] adicionar pairwise específico P13+/P12+;
20. [ ] adicionar distribuição completa dos deltas de acertos.

## Fase 4 — robustez

21. [ ] comparar expanding × rolling windows;
22. [ ] testar decay temporal dentro do nested;
23. [ ] gerar reliability tables de Top1/Top2/Top3;
24. [ ] salvar CSVs de calibração;
25. [ ] criar/manter `output/experiments.csv` com commit do Git;
26. [ ] bootstrap pareado e IC95% de ΔP13+/ΔP12+ das melhores estratégias.

## Fase 5 — expansão completa

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
OracleDistribution / DistributionBacktest
      +
Opportunity Dataset
      +
DoubleValueModel
      +
JointDoubleOptimizer
      +
Nested Walk-Forward
      +
Regret / Oracle Capture
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