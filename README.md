# Loteca ML — Estratégia 9 Secos / 5 Duplos / 0 Triplos

Projeto para gerar **um único palpite final da Loteca**, buscando maximizar a chance de **13 ou 14 acertos**.

Estrutura obrigatória:

```text
14 jogos
9 secos
5 duplos
0 triplos
19 marcações
```

> A unidade de avaliação é o **ticket completo**. A métrica principal é **P(>=13)**, não accuracy isolada por partida.

---

# Dados e ranking

Arquivos principais:

```text
data/concursos_anteriores.csv
data/proximo_concurso.csv
```

Probabilidades:

```text
p(1) = vitória do mandante
p(X) = empate
p(2) = vitória do visitante
p(1) + p(X) + p(2) = 1
```

Ranking:

```text
Top1 = maior probabilidade
Top2 = segunda maior
Top3 = menor probabilidade
```

Desempate:

```text
1 > 2 > X
```

Base atual:

```text
448 concursos
30 concursos na janela inicial
418 concursos avaliados em walk-forward

Top1: 51.6741%
Top2: 26.5625%
Top3: 21.7634%
```

---

# Função objetivo

Ordem de comparação:

```text
1. P13+
2. número de 14
3. número de 13
4. P12+
5. número de 12
6. média de acertos
7. estabilidade
```

Brier, Log Loss, ECE, média, regret e win rates intermediários são diagnósticos. Uma estratégia só pode substituir o baseline se melhorar o **ticket fora da amostra**.

---

# Constraints

## Hard Constraint — Flamengo

Quando o **FLAMENGO/RJ** participar, sua vitória deve obrigatoriamente estar coberta:

```text
Flamengo mandante  → incluir 1
Flamengo visitante → incluir 2
```

## Soft Constraint — Palmeiras

Favorecer a exclusão da vitória do **PALMEIRAS/SP** apenas quando o custo probabilístico for pequeno.

```text
limiar atual = 0.03
```

A preferência do Palmeiras nunca pode violar Hard Constraints.

---

# Baseline operacional

Arquitetura atual:

```text
14 jogos
   ↓
DoubleAllocator
   ↓
5 jogos recebem duplo
   ↓
SecondMarkSelector
   ↓
Constraints
   ↓
Ticket
```

Políticas:

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

Estratégia operacional:

```text
allocator: uncertainty
second mark: top2_baseline
```

Nenhuma alternativa demonstrou ganho prospectivo suficiente em P13+ para substituir esse baseline.

---

# Backtest atual

418 concursos:

```text
gain / top2_probability
P13+ 1.4354% | P12+ 5.5024% | média 8.7273

uncertainty
P13+ 1.4354% | P12+ 5.9809% | média 8.7081

margin
P13+ 1.1962% | P12+ 5.5024% | média 8.7129

ratio
P13+ 1.4354% | P12+ 5.5024% | média 8.7010

hist_top1
P13+ 1.1962% | P12+ 5.2632% | média 8.5526

hist_top2
P13+ 1.1962% | P12+ 5.9809% | média 8.5861

exact
P13+ 1.4354% | P12+ 5.7416% | média 8.6962
```

Overlap com `uncertainty`:

```text
gain:             4.297 / 5
top2_probability: 4.297 / 5
ratio:            4.730 / 5
exact:            4.658 / 5
```

Leitura:

> Os allocators tradicionais escolhem quase os mesmos cinco jogos. Trocar apenas a heurística de alocação tende a produzir ganho marginal pequeno.

---

# Calibração e Top1

```text
Brier multiclass: 0.588667
Log Loss:         0.985938
ECE:              0.012009
```

Correções Top1:

```text
top1_residual:    48.40%
top1_lift:        48.59%
top1_reliability: 47.71%
p(top1_meta):     43.88%
```

Top1-meta:

```text
Brier baseline: 0.233746
Brier meta:     0.240623
```

Conclusão:

> As correções atuais do Top1 permanecem apenas como telemetria.

---

# SecondMarkSelector / Recovery

```text
742 disagreements
Top2 baseline: 368
recovery:      374
win rate:      50.40%
```

Nested:

```text
delta P13+: -0.24 p.p.
delta P12+: +2.63 p.p.
```

Conclusão:

> Recovery melhora P12+, mas piora P13+. `top2_baseline` permanece operacional.

---

# Oracle Decomposition

Diagnóstico retrospectivo:

```text
baseline
P13+ 1.44% | P12+ 5.98% | média 8.7081

allocator oracle
P13+ 11.00% | P12+ 30.86% | média 10.7129

selector oracle
P13+ 5.50% | P12+ 21.29% | média 10.1794

full oracle
P13+ 41.63% | P12+ 64.59% | média 12.0144
```

Principal leitura:

> O maior espaço de melhoria está na estrutura das marcações: onde proteger, qual rank usar e, no XYZ, quais Top1 abandonar.

---

# Espaço SAFE

Top1 permanece nos 14 jogos; as cinco marcas extras são distribuídas entre Top2 e Top3.

```text
14/5/0: P13+ 1.44% | P12+ 5.50% | média 8.7297
14/4/1: P13+ 0.48% | P12+ 5.98% | média 8.7273
14/3/2: P13+ 0.72% | P12+ 6.22% | média 8.7512
14/2/3: P13+ 1.20% | P12+ 5.74% | média 8.7679
14/1/4: P13+ 1.20% | P12+ 5.26% | média 8.7464
14/0/5: P13+ 1.67% | P12+ 5.26% | média 8.6842
```

```text
melhor P13+: 14/0/5
melhor P12+: 14/3/2
melhor média: 14/2/3
```

Não promover `14/0/5` sem NestedDistributionSelector e bootstrap pareado.

## OracleDistribution

```text
P13+: 41.39%
P12+: 63.88%
```

Comparação:

```text
OracleDistribution: 41.39%
OracleFull:         41.63%
```

> Quase todo o teto do OracleFull pode ser reproduzido escolhendo corretamente as cinco proteções sem remover Top1.

---

# Espaço XYZ

Definição:

```text
X = total de marcações Top1
Y = total de marcações Top2
Z = total de marcações Top3
X + Y + Z = 19
```

Centro:

```text
9/5/5
```

Ações permitidas por jogo:

```text
T1
T2
T3
T1T2
T1T3
T2T3
```

O DP preserva:

```text
9 secos
5 duplos
19 marcações
X/Y/Z exatos
Hard Constraint do Flamengo
```

Distribuições do raio 1:

```text
9/5/5
10/5/4
10/4/5
9/6/4
9/4/6
8/6/5
8/5/6
```

---

# XYZ operacional — função objetivo coverage_sum

```text
XYZ_09_05_05: P13+ 0.48% | P12+ 3.59% | média 8.2153
XYZ_10_05_04: P13+ 0.24% | P12+ 5.02% | média 8.4354
XYZ_10_04_05: P13+ 0.24% | P12+ 3.59% | média 8.3206
XYZ_09_06_04: P13+ 0.72% | P12+ 3.35% | média 8.2608
XYZ_09_04_06: P13+ 0.48% | P12+ 2.87% | média 8.1364
XYZ_08_06_05: P13+ 0.48% | P12+ 2.63% | média 8.0478
XYZ_08_05_06: P13+ 0.48% | P12+ 1.44% | média 7.9856
```

Comparação:

```text
best SAFE: 14/0/5
best XYZ:  XYZ_09_06_04

delta P13+: -0.96 p.p.
delta P12+: -1.91 p.p.
delta média: -0.4234
```

> O XYZ raio 1 perdeu para SAFE usando soma de cobertura probabilística. Isso não invalida o espaço XYZ; invalida a capacidade do otimizador atual de explorá-lo.

---

# XYZ Retrospective Frozen Selection

Escolha retrospectiva entre sete tickets já construídos por probabilidades:

```text
P13+: 0.96%
P12+: 7.18%
média: 8.9785
```

Esse diagnóstico **não é um oracle estrutural**.

---

# Actual Rank Profile — implementado

Perfil real dos ranks nos 418 concursos:

```text
média   T1/T2/T3 = 7.201 / 3.763 / 3.036
mediana T1/T2/T3 = 7 / 4 / 3
```

Perfis mais frequentes:

```text
7/4/3 → 31 concursos (7.42%)
8/3/3 → 28 concursos (6.70%)
6/5/3 → 24 concursos (5.74%)
8/4/2 → 21 concursos (5.02%)
7/3/4 → 18 concursos (4.31%)
9/3/2 → 18 concursos (4.31%)
```

Leitura:

> O concurso típico tem aproximadamente sete resultados Top1, quatro Top2 e três Top3. A principal dificuldade do XYZ é identificar antecipadamente **quais** Top1 falharão.

---

# TrueOracleXYZ — implementado

O TrueOracleXYZ usa resultados reais somente dentro do diagnóstico retrospectivo.

```text
XYZ_09_05_05: P13+ 88.04% | P12+ 96.89% | média 13.4569
XYZ_10_05_04: P13+ 84.69% | P12+ 97.13% | média 13.4163
XYZ_10_04_05: P13+ 81.10% | P12+ 93.78% | média 13.2703
XYZ_09_06_04: P13+ 85.65% | P12+ 97.61% | média 13.4402
XYZ_09_04_06: P13+ 77.99% | P12+ 92.82% | média 13.2010
XYZ_08_06_05: P13+ 83.01% | P12+ 94.50% | média 13.3469
XYZ_08_05_06: P13+ 79.67% | P12+ 93.06% | média 13.2536
```

Global:

```text
P13+: 96.89%
P12+: 99.52%
média: 13.8397
```

Usage:

```text
9/5/5:  258
10/5/4: 60
10/4/5: 0
9/6/4:  66
9/4/6:  29
8/6/5:  4
8/5/6:  1
```

> `96.89%` é **teto de representação**, não expectativa operacional. O oracle conhece os resultados e escolhe exatamente quais Top1 abandonar.

O raio 1 já contém enorme capacidade estrutural; **não abrir radius=2 agora**.

---

# XYZ Oracle Feasibility — implementado

```text
XYZ_09_05_05: feasible14 61.72% | feasible13+ 88.04%
XYZ_10_05_04: feasible14 60.77% | feasible13+ 84.69%
XYZ_10_04_05: feasible14 55.02% | feasible13+ 81.10%
XYZ_09_06_04: feasible14 61.24% | feasible13+ 85.65%
XYZ_09_04_06: feasible14 52.15% | feasible13+ 77.99%
XYZ_08_06_05: feasible14 58.85% | feasible13+ 83.01%
XYZ_08_05_06: feasible14 54.78% | feasible13+ 79.67%
```

Isso separa:

```text
capacidade da distribuição
vs
capacidade da seleção pré-jogo
```

---

# ExactXYZP13Optimizer — implementado

O DP XYZ atual maximiza:

```text
coverage_sum = soma das probabilidades cobertas
```

O objetivo real é:

```text
P(>=13) = P(14) + P(13)
```

Para cada ticket:

```text
q_i = probabilidade coberta no jogo i

P14 = produto(q_i)
P13 = soma_i [(1-q_i) × produto(q_j, j != i)]
```

Dois otimizadores explícitos estão disponíveis:

```text
XYZ_COVERAGE
XYZ_DIRECT_P13
```

`XYZ_DIRECT_P13` mantém, em cada estado estrutural do DP, a fronteira de
Pareto exata de `P(14)` e `P(13)`. Estados dominados são descartados, pois a
transição Poisson-binomial é monotônica nas duas probabilidades. Assim, o
ticket final maximiza diretamente `P(14) + P(13)` sem enumerar `6^14`
combinações e sem relaxar `9 secos / 5 duplos / 0 triplos` ou a cobertura da
vitória do Flamengo.

Telemetria:

```text
[XYZ OBJECTIVE COMPARISON]
distribution | coverage historical P13+ | direct historical P13+ | delta
```

Também são registrados o ganho médio modelado de `P13+` e a verificação de
que o objetivo direto nunca é pior que `coverage_sum` na probabilidade
pré-jogo do próprio ticket.

Registrar também a probabilidade teórica prevista pelo modelo:

```text
coverage_model_P13+
direct_model_P13+
```

Isso permite distinguir ganho matemático previsto de ganho histórico real.

---

# Diagnóstico prioritário — Top1 Miss Capture

Antes de criar um modelo ML complexo, medir quanto o próprio `1 - pTop1` consegue localizar os erros do Top1.

Para cada concurso, ordenar os 14 jogos por:

```text
miss_score = 1 - pTop1
```

Telemetria:

```text
[TOP1 MISS CAPTURE]
k=1 → 267 / 2842 (9.39%)
k=3 → 773 / 2842 (27.20%)
k=5 → 1253 / 2842 (44.09%)
k=7 → 1683 / 2842 (59.22%)
```

Implementado como diagnóstico retrospectivo no mesmo recorte de avaliação
walk-forward. O ranking de candidatos usa exclusivamente `1 - pTop1`; o
resultado real entra apenas no cálculo de `captured_misses / total_top1_misses`.
O diagnóstico não altera o ticket operacional.

Objetivo:

> Saber se as probabilidades atuais já conseguem localizar os Top1 frágeis, mesmo que o DP esteja usando essa informação de forma inadequada.

---

# Diagnóstico prioritário — Top1 Drop Oracle Capture

Comparar os candidatos pré-jogo de abandono de Top1 com os Top1 efetivamente abandonados pelo TrueOracleXYZ.

```text
[TOP1 DROP ORACLE CAPTURE]
top 1 candidato: ...
top 3 candidatos: ...
top 5 candidatos: ...
top 7 candidatos: ...
```

Resultado atual:

```text
k=1 → 199 / 2035 (9.78%)
k=3 → 557 / 2035 (27.37%)
k=5 → 912 / 2035 (44.82%)
k=7 → 1222 / 2035 (60.05%)
```

Esse diagnóstico mede quanta informação útil existe para aproximar o comportamento do oracle sem utilizar resultado futuro.

Implementado com o TrueOracleXYZ de raio 1. Para cada concurso, o ticket oracle
é construído apenas na camada `diagnostic_only`; em seguida, os Top1 abandonados
são comparados com a lista pré-jogo ordenada por `1 - pTop1`. São reportados o
numerador, o total de abandonos e a taxa de captura, evitando que percentuais
com denominadores diferentes sejam confundidos.

---

# RankReplacement baselines

Depois de decidir abandonar Top1, testar primeiro regras simples para escolher Top2 ou Top3:

```text
always_top2
always_top3
probability_conditional
recovery_conditional
```

Avaliar apenas nos casos de Top1 miss:

```text
replacement_accuracy
top2_wins
top3_wins
```

Somente depois desses benchmarks criar um `RankReplacementModel` aprendido.

---

# NestedDistributionSelector SAFE

Validar prospectivamente:

```text
14/5/0
14/4/1
14/3/2
14/2/3
14/1/4
14/0/5
```

Fluxo:

```text
histórico até N
      ↓
selecionar somente com passado
      ↓
congelar distribuição
      ↓
aplicar em N+1
```

Somente esse teste pode promover `14/0/5` ou outra distribuição SAFE.

---

# Pairwise e bootstrap

Comparações prioritárias:

```text
XYZ_DIRECT_P13 vs XYZ_COVERAGE
14/0/5 vs 14/5/0
14/0/5 vs uncertainty
best XYZ vs best SAFE
```

Pairwise tail-aware:

```text
wins / ties / losses
13+ exclusivo A / ambos / exclusivo B
12+ exclusivo A / ambos / exclusivo B
```

Bootstrap:

```text
unidade = concurso inteiro
delta P13+
delta P12+
delta média
IC95%
```

---

# Modelos aprendidos — somente depois dos benchmarks

## Top1DropModel

Target:

```text
top1_miss = 1 se Top1 falhou
```

Features candidatas:

```text
p_top1
p_top2
p_top3
margin_12
gap_23
entropy
ratio_top2_top1
ratio_top3_top1
posição
perfil do concurso
```

Uso:

> Ranqueia quais Top1 são melhores candidatos a serem sacrificados quando a distribuição XYZ exigir isso.

## RankReplacementModel

Pergunta:

```text
Top2 ou Top3?
```

Deve vencer os baselines simples em walk-forward antes de entrar no pipeline.

---

# JointMarkAllocator

Tratar diretamente oportunidades:

```text
(game_i, Top2)
(game_i, Top3)
```

Selecionar exatamente cinco, no máximo uma por jogo.

Baseline:

```text
score Top2 = pTop2
score Top3 = pTop3
```

Essa linha continua relevante porque OracleDistribution e OracleFull ficaram praticamente empatados.

---

# Opportunity Dataset / DoubleValueModel

Uma linha por:

```text
concurso × jogo × marca candidata
```

Features:

```text
p_top1
p_top2
p_top3
gap_12
gap_23
entropy
ratio_top2_top1
ratio_top3_top1
posição
perfil do concurso
```

Target:

```text
extra_mark_hit = 1
```

quando a marca adicional recupera um erro do Top1.

---

# Robustez

Implementações futuras:

```text
rolling 50
rolling 100
rolling 200
expanding
stability por era
decay temporal
Stability / Churn
controle de múltiplos testes
output/experiments.csv
```

Quanto maior o search space, maior o risco de `winner's curse`.

---

# Testes obrigatórios

```text
9 secos / 5 duplos / 0 triplos
19 marcações
X+Y+Z=19 no XYZ
Flamengo sempre coberto
sem informação futura no pipeline operacional
TrueOracleXYZ somente diagnostic_only
P13+ exato reproduzível
bootstrap por concurso inteiro
train_contest < test_contest
```

---

# Execução

```powershell
python main.py
python -m unittest discover -v
```

---

# Roadmap

## Concluído

- [x] pipeline de constraints;
- [x] invariantes 9/5/0;
- [x] walk-forward;
- [x] métricas P13+/P12+;
- [x] calibração;
- [x] allocators e pairwise inicial;
- [x] Top1 diagnostics;
- [x] recovery + nested recovery;
- [x] OracleAllocator / Selector / Full;
- [x] DistributionBacktest SAFE;
- [x] OracleDistribution;
- [x] núcleo XYZ via DP;
- [x] XYZ raio 1;
- [x] XYZ vs SAFE;
- [x] Frozen Selection XYZ;
- [x] TrueOracleXYZByDistribution;
- [x] TrueOracleXYZ + Usage;
- [x] Actual Rank Profile;
- [x] XYZ Oracle Feasibility.

## Fase 1 — alinhar o objetivo ao P13+

1. [ ] cálculo exato de P13+ para estados/tickets;
2. [ ] `ExactXYZP13Optimizer`;
3. [ ] `XYZ_COVERAGE × XYZ_DIRECT_P13`;
4. [ ] telemetria `model_P13+ × historical_P13+`;
5. [ ] pairwise tail-aware;
6. [ ] bootstrap pareado.

## Fase 2 — medir o sinal disponível para abandonar Top1

7. [x] `Top1 Miss Capture`;
8. [x] `Top1 Drop Oracle Capture`;
9. [ ] Structural Gap / Oracle Capture baseado em média;
10. [ ] segmentação por força de Top1, margem e entropia.

## Fase 3 — validar SAFE

11. [ ] `NestedDistributionSelector`;
12. [ ] bootstrap `14/0/5 vs 14/5/0`;
13. [ ] OracleDistribution Usage;
14. [ ] regret por distribuição SAFE fixa.

## Fase 4 — aprender seleção XYZ

15. [ ] RankReplacement baselines;
16. [ ] dataset Top1DropModel;
17. [ ] walk-forward Top1DropModel;
18. [ ] RankReplacementModel;
19. [ ] comparar arquitetura modular com `XYZ_DIRECT_P13`;
20. [ ] somente então `NestedXYZDistributionSelector`.

## Fase 5 — otimização conjunta

21. [ ] JointMarkAllocator;
22. [ ] `joint_probability`;
23. [ ] Opportunity Dataset;
24. [ ] DoubleValueModel;
25. [ ] `joint_learned`;
26. [ ] nested walk-forward.

## Fase 6 — robustez final

27. [ ] rolling 50/100/200 vs expanding;
28. [ ] stability por era;
29. [ ] decay temporal;
30. [ ] Stability / Churn;
31. [ ] controle de múltiplos testes;
32. [ ] `output/experiments.csv`;
33. [ ] bootstrap final.

## Radius=2

```text
NÃO ABRIR AGORA
```

O raio 1 já possui enorme capacidade estrutural. Expandir antes de melhorar a seleção pré-jogo aumenta apenas o risco de overfitting.

---

# Critério de promoção

Uma estratégia só pode substituir o baseline quando:

```text
melhorar P13+ fora da amostra
↓
ser escolhida sem olhar o período de teste
↓
apresentar resultado pareado favorável
↓
apresentar bootstrap/IC compatível com ganho real
↓
manter estabilidade temporal
↓
respeitar todas as Hard Constraints
```

Para XYZ:

```text
TrueOracleXYZ prova capacidade, não previsibilidade
↓
novo otimizador precisa capturar parte do headroom usando apenas pré-jogo
↓
Direct-P13 precisa vencer Coverage fora da amostra
↓
modelos aprendidos precisam vencer benchmarks simples
↓
NestedXYZ só entra quando existir candidato operacional competitivo
```

---

# Princípio geral

```text
Baseline operacional
      +
SAFE / OracleDistribution
      +
XYZ raio 1 / TrueOracleXYZ
      ↓
CAPACIDADE ESTRUTURAL CONFIRMADA
      ↓
Actual Rank Profile / Oracle Feasibility
      +
ExactXYZP13Optimizer
      +
Top1 Miss Capture / Drop Oracle Capture
      +
Pairwise / Bootstrap
      +
Nested SAFE
      +
Top1DropModel / RankReplacementModel
      +
JointMarkAllocator / DoubleValueModel
      +
NestedXYZ somente com candidato competitivo
      +
Robustez temporal
      ↓
PALPITE FINAL
```

> **Não promover a melhor regra observada; promover apenas a regra que continuar melhor quando escolhida usando somente o passado e avaliada prospectivamente.**
