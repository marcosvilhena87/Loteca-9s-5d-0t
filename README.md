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

Ranking por jogo:

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

Brier, Log Loss, ECE, regret, overlap, win rates, Capture@K, Lift@K, NDCG, MAP e métricas intermediárias são **diagnósticos**.

Uma estratégia só pode substituir o baseline se melhorar o **ticket completo fora da amostra**.

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

Políticas avaliadas:

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
P13+ 1.6746% | P12+ 5.5024% | média 8.7297

uncertainty
P13+ 1.6746% | P12+ 5.9809% | média 8.7105

margin
P13+ 1.4354% | P12+ 5.5024% | média 8.7153

ratio
P13+ 1.6746% | P12+ 5.5024% | média 8.7033

hist_top1
P13+ 1.4354% | P12+ 5.2632% | média 8.5526

hist_top2
P13+ 1.4354% | P12+ 5.9809% | média 8.5861

exact
P13+ 1.6746% | P12+ 5.7416% | média 8.6962
```

Overlap com `uncertainty`:

```text
gain:             4.299 / 5
top2_probability: 4.299 / 5
ratio:            4.730 / 5
exact:            4.658 / 5
```

Pairwise inicial:

```text
gain vs uncertainty
62 vitórias | 302 empates | 54 derrotas | delta médio +0.0191
```

> Os allocators tradicionais escolhem quase os mesmos cinco jogos. O gargalo principal não parece ser encontrar mais uma heurística global parecida, mas discriminar melhor **quais jogos merecem proteção em cada concurso**.

---

# Calibração e Top1

```text
Brier multiclass: 0.588687
Log Loss:         0.985982
ECE:              0.012141
```

Correções Top1:

```text
top1_residual:    48.56%
top1_lift:        48.70%
top1_reliability: 47.59%
p(top1_meta):     43.88%
```

Top1-meta:

```text
Brier baseline: 0.233761
Brier meta:     0.240627
```

> As correções atuais permanecem apenas como telemetria.

---

# SecondMarkSelector / Recovery

```text
743 disagreements
Top2 baseline: 369
recovery:      374
win rate:      50.34%
```

Thresholds:

```text
0.00 → 50.34%
0.02 → 49.55%
0.05 → 52.36%
0.10 → 52.85%
0.15 → 52.77%
```

Nested:

```text
delta P13+: -0.24 p.p.
delta P12+: +2.63 p.p.
thresholds escolhidos: 0.05 em 376 concursos; 0.10 em 42
```

> Recovery melhora P12+, mas piora P13+. `top2_baseline` permanece operacional.

---

# Oracle Decomposition

```text
baseline
P13+ 1.67% | P12+ 5.98% | média 8.7105

allocator oracle
P13+ 11.24% | P12+ 30.86% | média 10.7129

selector oracle
P13+ 5.50% | P12+ 21.29% | média 10.1794

full oracle
P13+ 41.63% | P12+ 64.59% | média 12.0144
```

Regret:

```text
allocator: média 2.0024 | zero 8.85% | 2+ 67.46% | máximo 5
selector:  média 1.4689 | zero 17.94% | 2+ 44.26% | máximo 4
full:      média 3.3038 | zero 0.96% | 2+ 95.22% | máximo 5
```

> O maior espaço de melhoria está em **onde proteger, qual rank usar e quais Top1 abandonar**.

---

# Espaço SAFE

Top1 permanece nos 14 jogos; as cinco marcas extras são distribuídas entre Top2 e Top3.

```text
14/5/0: P13+ 1.67% | P12+ 5.50% | média 8.7321
14/4/1: P13+ 0.24% | P12+ 6.22% | média 8.7297
14/3/2: P13+ 0.48% | P12+ 6.46% | média 8.7512
14/2/3: P13+ 0.96% | P12+ 5.98% | média 8.7679
14/1/4: P13+ 0.96% | P12+ 5.26% | média 8.7440
14/0/5: P13+ 1.67% | P12+ 5.26% | média 8.6818
```

```text
melhor P13+: empate 14/5/0 e 14/0/5
melhor P12+: 14/3/2
melhor média: 14/2/3
```

Pelo critério hierárquico, `14/5/0` permanece à frente de `14/0/5` porque empata em P13+ e tem P12+ superior.

## OracleDistribution

```text
P13+: 41.39%
P12+: 63.88%
```

```text
OracleDistribution: 41.39%
OracleFull:         41.63%
```

> Quase todo o teto do OracleFull pode ser reproduzido escolhendo corretamente as cinco proteções sem remover Top1.

---

# Espaço XYZ

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

Ações permitidas:

```text
T1
T2
T3
T1T2
T1T3
T2T3
```

Distribuições de raio 1:

```text
9/5/5
10/5/4
10/4/5
9/6/4
9/4/6
8/6/5
8/5/6
```

## XYZ coverage_sum

```text
XYZ_09_05_05: P13+ 0.48% | P12+ 3.59% | média 8.2177
XYZ_10_05_04: P13+ 0.24% | P12+ 5.02% | média 8.4378
XYZ_10_04_05: P13+ 0.24% | P12+ 3.59% | média 8.3254
XYZ_09_06_04: P13+ 0.72% | P12+ 3.35% | média 8.2608
XYZ_09_04_06: P13+ 0.48% | P12+ 2.87% | média 8.1388
XYZ_08_06_05: P13+ 0.48% | P12+ 2.63% | média 8.0502
XYZ_08_05_06: P13+ 0.48% | P12+ 1.44% | média 7.9856
```

Comparação:

```text
best SAFE: 14/5/0
best XYZ:  XYZ_09_06_04

delta P13+: -0.96 p.p.
delta P12+: -2.15 p.p.
delta média: -0.4713
```

> O raio 1 já possui enorme capacidade estrutural. O problema é seleção, não falta de combinações.

---

# Actual Rank Profile

```text
média   T1/T2/T3 = 7.201 / 3.763 / 3.036
mediana T1/T2/T3 = 7 / 4 / 3
```

Perfis mais frequentes:

```text
7/4/3 → 30 concursos (7.18%)
8/3/3 → 28 concursos (6.70%)
6/5/3 → 24 concursos (5.74%)
8/4/2 → 21 concursos (5.02%)
7/3/4 → 19 concursos (4.55%)
9/3/2 → 18 concursos (4.31%)
```

---

# TrueOracleXYZ

```text
P13+: 96.89%
P12+: 99.52%
média: 13.8397
```

> `96.89%` é **teto de representação**, não expectativa operacional. O oracle conhece os resultados e escolhe quais Top1 abandonar.

**Não abrir `radius=2` agora.**

---

# ExactXYZP13Optimizer

Objetivo real:

```text
P(>=13) = P(14) + P(13)
```

Otimizadores:

```text
XYZ_COVERAGE
XYZ_DIRECT_P13
```

Resultado histórico:

```text
                  coverage   direct   delta P13+
XYZ_09_05_05       0.48%     0.96%     +0.48 p.p.
XYZ_10_05_04       0.24%     0.72%     +0.48 p.p.
XYZ_10_04_05       0.24%     0.48%     +0.24 p.p.
XYZ_09_06_04       0.72%     0.96%     +0.24 p.p.
XYZ_09_04_06       0.48%     0.48%     +0.00 p.p.
XYZ_08_06_05       0.48%     0.48%     +0.00 p.p.
XYZ_08_05_06       0.48%     0.72%     +0.24 p.p.
```

> O objetivo direto melhorou P13+ histórico em cinco das sete distribuições, mas ainda precisa de comparação pareada e bootstrap.

---

# Top1 Miss / Oracle Drop Capture

## Top1 Miss Capture

```text
k=1 → 267 / 2842 (9.39%)
k=3 → 773 / 2842 (27.20%)
k=5 → 1253 / 2842 (44.09%)
k=7 → 1683 / 2842 (59.22%)
```

## Top1 Drop Oracle Capture

```text
k=1 → 199 / 2035 (9.78%)
k=3 → 557 / 2035 (27.37%)
k=5 → 912 / 2035 (44.82%)
k=7 → 1222 / 2035 (60.05%)
```

Lift aproximado do Oracle Drop:

```text
k=1 → 1.37x
k=3 → 1.28x
k=5 → 1.25x
k=7 → 1.20x
```

```text
capacidade estrutural do XYZ: muito alta
sinal pré-jogo para Top1 frágil: existe
concentração desse sinal em poucos jogos: ainda insuficiente
```

---

# Top1FragilityBenchmark — implementado

Capture@5:

```text
score       Top1MissCapture  OracleDropCapture
1-pTop1          44.09%            44.82%
1-margin12       44.19%            45.06%
entropy          43.84%            44.47%
ratio2           43.95%            44.77%
ratio3           43.63%            44.08%
gap23            41.20%            41.97%
ensemble         43.84%            44.62%
```

> `1-margin12` lidera, mas supera `1-pTop1` em apenas **+0.24 p.p.**. Nenhum score simples resolveu o problema global.

---

# Top1FragilitySegments — implementado

Bins de `pTop1`:

```text
0.33–0.40
0.40–0.45
0.45–0.50
0.50–0.55
0.55–0.60
0.60+
```

Resultado atual:

```text
segmento    n     miss      oracle drop   Capture@5   Lift@5
0.33-0.40   1244  63.34%    43.97%        97.62%      1.01x
0.40-0.45   1176  55.19%    40.22%        59.62%      1.03x
0.45-0.50    966  53.21%    38.10%        20.38%      1.21x
0.50-0.55    722  45.71%    33.66%         6.17%      1.35x
0.55-0.60    650  37.85%    27.08%         2.84%      1.68x
0.60+       1094  28.79%    20.84%         0.44%      1.20x
```

Leitura importante:

> Capture alto em um segmento não significa necessariamente alta discriminação. O bin `0.33–0.40` tem Capture@5 de 97.62%, mas Lift@5 de apenas 1.01x porque quase todo esse segmento já cai naturalmente no top-5 de fragilidade.

O próximo passo é descobrir **qual score funciona melhor dentro de cada regime**.

---

# FragilityScoreBySegment — próxima prioridade

Para cada faixa de `pTop1`, comparar:

```text
1-pTop1
1-margin12
entropy
ratio2
ratio3
gap23
consensus
```

Saída desejada:

```text
segmento | best_score | DropCapture@K | Lift@K | n
```

Objetivo:

> Criar primeiro uma regra condicional simples antes de recorrer a ML.

---

# Matriz pTop1 × margin12

Cruzar duas dimensões:

```text
força absoluta do favorito → pTop1
competitividade Top1/Top2  → margin12
```

Exemplo estrutural:

```text
                 margin baixo   margin médio   margin alto
pTop1 baixo
pTop1 médio
pTop1 alto
```

Por célula:

```text
n
miss_rate
oracle_drop_rate
Top2_rate
Top3_rate
best_fragility_score
```

---

# PairwiseFragilityRanking

Comparações prioritárias:

```text
margin vs p
entropy vs p
ratio2 vs p
consensus vs p
```

Avaliar apenas concursos em que os top-K diferem:

```text
concursos com diferença
oracle drops extras A
oracle drops extras B
empates
delta médio de captura
```

---

# Ranking intra-concurso

O problema real é:

```text
entre os 14 jogos deste concurso,
quais Top1 são mais sacrificáveis?
```

Features relativas:

```text
rank_p_top1
rank_margin
rank_entropy
rank_ratio2

p_top1 - mean_p_top1_contest
margin - mean_margin_contest
entropy - mean_entropy_contest

zscore_p_top1
zscore_margin
zscore_entropy
```

Métricas diagnósticas:

```text
OracleDropCapture@K
NDCG@K
MAP@K
MRR
```

---

# FragilityConsensus

O ensemble por média não ajudou. Testar consenso ordinal:

```text
+1 se top-K por 1-pTop1
+1 se top-K por margin
+1 se top-K por entropy
+1 se top-K por ratio2
```

```text
consensus_score = número de rankings que consideram o jogo frágil
```

---

# Diagnóstico histórico de posições

Um teste retrospectivo externo ao pipeline operacional avaliou as **2.002 combinações possíveis** de escolher cinco posições do ranking de `pTop1` para receber Top1+Top2.

Principais achados históricos:

```text
melhor para 14:   6-7-11-13-14
melhor grupo 13+: inclui 10-11-12-13-14 e outras combinações
melhor para 12+:  2-8-11-12-14
maior média:       6-8-10-11-13
```

A posição 11 apareceu repetidamente entre configurações fortes e teve incidência histórica elevada de Top2 vencedor.

Esses resultados são **retrospectivos** e não justificam uma regra fixa.

> A utilidade desse diagnóstico é sugerir que a posição relativa do jogo contém informação que pode entrar como **feature/prior fraco**.

---

# HistoricalPositionPrior

Transformar posição histórica em feature, nunca em regra:

```text
position_top1_miss_rate
position_top2_rate
position_top3_rate
```

O prior deve ser calculado somente com concursos anteriores ao teste:

```text
histórico < N
      ↓
calcular taxas por posição
      ↓
aplicar no concurso N
```

Preferir shrinkage para a média global:

```text
position_prior = BayesianShrinkage(position_rate, global_rate)
```

Uso futuro:

```text
Top1DropModel
RankReplacementModel
JointMarkAllocator
```

> Posição 11 não significa “sempre duplo”. Significa apenas que a posição pode carregar um pequeno prior adicional se o efeito sobreviver ao walk-forward.

---

# PositionStability

Repetir diagnósticos de posição em diferentes janelas:

```text
rolling 100
rolling 200
primeiras / médias / últimas eras
expanding
```

Métricas:

```text
Top1 miss rate por posição
Top2 hit rate por posição
Top3 hit rate por posição
frequência da posição entre top-N combinações
churn entre janelas
```

Objetivo:

> Distinguir um prior estrutural de um artefato de composição histórica.

---

# PositionCombinationDiagnostic

Formalizar no repositório o teste exaustivo de `C(14,5)=2002` combinações como diagnóstico **não operacional**.

Saídas:

```text
best_for_14
best_for_P13+
best_for_P12+
best_for_mean
position_frequency_in_top_10
position_frequency_in_top_50
position_frequency_in_top_100
```

Mais importante que a combinação campeã é verificar **quais posições aparecem repetidamente entre muitas soluções boas**.

---

# DynamicZoneAllocator

Testar uma hipótese intermediária entre regra fixa e busca totalmente livre:

```text
core_safe = posições 1..K
allocation_zone = posições K+1..14
```

Valores iniciais:

```text
K = 4 / 5 / 6 / 7 / 8 / 9
```

O allocator continua escolhendo dinamicamente cinco jogos dentro da zona.

`K` deve ser escolhido apenas com dados passados e aplicado no concurso seguinte.

Objetivo:

> Testar se existe um núcleo de Top1 suficientemente forte para ser quase sempre seco, sem congelar quais posições receberão duplo.

---

# Position × Probability Interaction

Criar features de interação:

```text
position × pTop1
position × margin12
position × entropy
position × ratio2
```

A posição isolada não deve dominar o modelo; o objetivo é capturar regimes como:

```text
posição intermediária + Top1 moderado + margem pequena
```

---

# RankReplacementBenchmark

Depois de decidir abandonar Top1:

> Top2 ou Top3?

Baselines:

```text
always_top2
always_top3
higher_probability
ratio_conditional
gap_conditional
recovery_conditional
historical_position_replacement
```

Thresholds iniciais:

```text
ratio3: 0.70 / 0.75 / 0.80 / 0.85 / 0.90 / 0.95
gap23:  0.01 / 0.02 / 0.03 / 0.05 / 0.08 / 0.10
```

Todo threshold e todo prior devem ser escolhidos somente com passado.

---

# NestedDistributionSelector SAFE

Fluxo:

```text
histórico até N-1
      ↓
avaliar 14/5/0 ... 14/0/5
      ↓
escolher distribuição
      ↓
congelar
      ↓
aplicar em N
```

Pergunta:

> É possível prever quando Top3 merece receber mais das cinco proteções?

---

# ObjectiveConflictReport

O diagnóstico de posições mostrou que a configuração que maximiza 14 não é necessariamente a que maximiza P13+, P12+ ou média.

Criar relatório explícito:

```text
strategy
rank_P14
rank_P13+
rank_P12+
rank_mean
```

Objetivo:

> Impedir promoção indevida baseada em uma métrica secundária visualmente chamativa.

---

# ParetoStrategyFrontier

Comparar estratégias por:

```text
P13+
P12+
mean
stability
```

Eliminar estratégias dominadas antes de testes mais caros.

---

# TailAwarePairwise

Toda comparação relevante deve registrar:

```text
A_only_14
B_only_14
both_14

A_only_13plus
B_only_13plus
both_13plus
neither_13plus

A_only_12plus
B_only_12plus
both_12plus

wins
ties
losses
delta_mean
```

---

# Bootstrap

## Pareado por concurso

```text
delta P13+
delta P12+
delta média
IC95%
P(delta P13+ > 0)
```

## Block bootstrap

```text
block_size = 5
block_size = 10
block_size = 20 concursos
```

---

# Top1DropModel — somente depois dos benchmarks

Primeiro modelo recomendado: **regressão logística**.

Target preferencial:

```text
oracle_drop = 1 se TrueOracleXYZ abandona Top1 naquele jogo
```

Features:

```text
p_top1
p_top2
p_top3
margin_12
gap_23
entropy
ratio_top2_top1
ratio_top3_top1

rank_p_top1
rank_margin
rank_entropy
position_prior
position × pTop1
position × margin12
```

O modelo precisa vencer prospectivamente:

```text
1-pTop1
1-margin12
melhor score por segmento
melhor ranking intra-concurso
```

---

# DropValue / DecisionValue

Nem todo Top1 com alta chance de erro merece ser abandonado.

```text
DropValue_i =
P13+(melhor ticket permitindo drop no jogo i)
-
P13+(melhor ticket mantendo Top1 no jogo i)
```

Evolução futura:

```text
DecisionScore_i = expected_delta_P13+ + λ * position_prior
```

`λ` deve ser pequeno e escolhido somente no treino.

---

# CounterfactualAllocatorTable

Diagnóstico por concurso:

```text
jogo
rank_position
pTop1
pTop2
margin
entropy
position_prior
fragility_rank
oracle_drop
double_gain
drop_value
hit_delta
```

---

# JointMarkAllocator

Tratar diretamente oportunidades:

```text
(game_i, Top2)
(game_i, Top3)
```

Objetivo final:

```text
selecionar as marcações que maximizam P13+
do ticket completo
```

---

# Opportunity Dataset / DoubleValueModel

Uma linha por:

```text
concurso × jogo × marca candidata
```

Features candidatas:

```text
p_top1
p_top2
p_top3
gap_12
gap_23
entropy
ratio_top2_top1
ratio_top3_top1
rank_position
position_prior
perfil do concurso
```

Target inicial:

```text
extra_mark_hit = 1
```

Target futuro preferível:

```text
delta_ticket_P13+
```

---

# CLI diagnostics

A saída de `main.py` já inclui:

```text
[TOP1 MISS CAPTURE]
[TOP1 DROP ORACLE CAPTURE]
[FRAGILITY BENCHMARK @5]
[FRAGILITY SEGMENTS pTop1]
```

Próximos resumos desejáveis:

```text
[FRAGILITY SCORE BY SEGMENT]
[POSITION PRIOR]
[POSITION STABILITY]
[OBJECTIVE CONFLICT]
```

---

# ExperimentRegistry

Registrar execuções em:

```text
output/experiments.csv
```

Campos mínimos:

```text
experiment_id
timestamp
git_commit
dataset_hash
model_version
strategy
parameters
test_start
test_end
P13+
P12+
mean
Capture@5
Lift@5
bootstrap_low
bootstrap_high
```

---

# DatasetFingerprint

Todo resultado persistido deve identificar exatamente a base:

```text
dataset_rows
contests
first_contest
last_contest
dataset_sha256
model_version
git_commit
trained_at
```

Persistir em:

```text
model.json
logs
experiments.csv
```

---

# Robustez temporal

Comparar:

```text
rolling 50
rolling 100
rolling 200
expanding
leave-one-era-out
decay temporal
Stability / Churn
PositionStability
```

---

# Controle de múltiplos testes

Quanto maior o search space, maior o risco de `winner's curse`.

Obrigatório antes de promoção:

```text
registro de todos os experimentos
holdout final protegido
bootstrap pareado
block bootstrap
limite de decisões no mesmo recorte
controle de múltiplas comparações quando aplicável
```

---

# Testes obrigatórios

```text
9 secos / 5 duplos / 0 triplos
19 marcações
X+Y+Z=19 no XYZ
Flamengo sempre coberto
sem informação futura no pipeline operacional
TrueOracleXYZ somente diagnostic_only
position priors calculados somente com passado
P13+ exato reproduzível
train_contest < test_contest
DatasetFingerprint reproduzível
mesmo seed → mesmo resultado quando aplicável
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

- [x] pipeline de constraints e invariantes 9/5/0;
- [x] walk-forward e métricas P13+/P12+;
- [x] calibração e allocators;
- [x] recovery + nested recovery;
- [x] OracleAllocator / Selector / Full;
- [x] SAFE + OracleDistribution;
- [x] XYZ raio 1 + TrueOracleXYZ;
- [x] Actual Rank Profile + Oracle Feasibility;
- [x] ExactXYZP13Optimizer;
- [x] Top1 Miss Capture / Drop Oracle Capture;
- [x] Top1FragilityBenchmark;
- [x] FragilityBenchmark no CLI;
- [x] Top1FragilitySegments por pTop1.

## Fase 1 — validar objetivo e comparações

1. [ ] TailAwarePairwise `XYZ_DIRECT_P13 vs XYZ_COVERAGE`;
2. [ ] bootstrap pareado;
3. [ ] block bootstrap;
4. [ ] estabilidade do ganho por era.

## Fase 2 — fragilidade condicional

5. [ ] `FragilityScoreBySegment`;
6. [ ] matriz `pTop1 × margin12`;
7. [ ] segmentação por entropy/ratios/gap23;
8. [ ] `PairwiseFragilityRanking`;
9. [ ] ranking intra-concurso;
10. [ ] `FragilityConsensus`.

## Fase 3 — posição histórica como prior

11. [ ] `PositionCombinationDiagnostic` C(14,5);
12. [ ] frequência de posição entre top-10/top-50/top-100;
13. [ ] `HistoricalPositionPrior` walk-forward;
14. [ ] shrinkage das taxas por posição;
15. [ ] `PositionStability`;
16. [ ] `DynamicZoneAllocator`;
17. [ ] interações `position × probability`;
18. [ ] `ObjectiveConflictReport`;
19. [ ] `ParetoStrategyFrontier`.

## Fase 4 — replacement e SAFE

20. [ ] `RankReplacementBenchmark`;
21. [ ] `historical_position_replacement`;
22. [ ] thresholds nested Top2/Top3;
23. [ ] `NestedDistributionSelector SAFE`;
24. [ ] bootstrap `14/0/5 vs 14/5/0`;
25. [ ] OracleDistribution Usage / regret SAFE.

## Fase 5 — aprender ranking e valor da decisão

26. [ ] dataset `Top1DropModel`;
27. [ ] regressão logística walk-forward;
28. [ ] Top1DropModel vs benchmarks simples;
29. [ ] `RankReplacementModel`;
30. [ ] `DropValue / DecisionValue`;
31. [ ] `CounterfactualAllocatorTable`;
32. [ ] `JointMarkAllocator`;
33. [ ] Opportunity Dataset / DoubleValueModel;
34. [ ] nested walk-forward;
35. [ ] somente então `NestedXYZDistributionSelector`.

## Fase 6 — reprodutibilidade

36. [ ] `DatasetFingerprint`;
37. [ ] `ExperimentRegistry`;
38. [ ] persistir parâmetros, hashes e versão;
39. [ ] CLI diagnostics padronizado.

## Fase 7 — robustez final

40. [ ] rolling 50/100/200 vs expanding;
41. [ ] leave-one-era-out;
42. [ ] decay temporal;
43. [ ] Stability / Churn;
44. [ ] controle de múltiplos testes;
45. [ ] bootstrap final;
46. [ ] holdout final protegido.

---

# Radius=2

```text
NÃO ABRIR AGORA
```

Também não priorizar agora:

```text
regras fixas como 6-7-11-13-14
deep learning
grande grid search
XGBoost/LightGBM hiperotimizado
mais dezenas de distribuições XYZ
```

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
respeitar Hard Constraints
↓
ser reproduzível por dataset/model/git fingerprint
```

Para priors de posição:

```text
posição histórica sugere hipótese
↓
prior calculado somente com passado
↓
efeito precisa sobreviver a PositionStability
↓
prior precisa melhorar seleção/ticket fora da amostra
↓
nunca virar regra fixa por posição
```

Para XYZ:

```text
TrueOracleXYZ prova capacidade, não previsibilidade
↓
Direct-P13 precisa vencer Coverage fora da amostra
↓
fragility/ranking precisa vencer benchmarks simples
↓
modelos precisam vencer regras condicionais simples
↓
NestedXYZ só entra com candidato operacional competitivo
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
Top1FragilityBenchmark / Segments
      ↓
SCORES GLOBAIS QUASE EMPATADOS
      ↓
FragilityScoreBySegment
      +
Ranking intra-concurso
      +
HistoricalPositionPrior
      +
PositionStability
      ↓
RankReplacementBenchmark
      +
Nested SAFE
      ↓
Top1DropModel / RankReplacementModel
      ↓
DropValue / JointMarkAllocator
      ↓
NestedXYZ somente com candidato competitivo
      +
ExperimentRegistry / DatasetFingerprint
      +
Robustez temporal / holdout final
      ↓
PALPITE FINAL
```

> **O projeto não precisa agora de mais espaço combinatório; precisa aprender a navegar muito melhor o espaço que já possui.**

> **Posição histórica pode ser prior; não pode virar regra fixa sem validação prospectiva.**

> **Não promover a melhor regra observada; promover apenas a regra que continuar melhor quando escolhida usando somente o passado e avaliada prospectivamente.**