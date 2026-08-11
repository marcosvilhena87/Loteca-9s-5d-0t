# Loteca ML — Estratégia 9 Secos / 5 Duplos / 0 Triplos

Projeto para gerar **um único palpite final da Loteca**, com foco em maximizar a chance de atingir **13 ou 14 acertos**, respeitando sempre:

- **9 secos**;
- **5 duplos**;
- **0 triplos**;
- **19 marcações**;
- **14 jogos por concurso**.

> A unidade de avaliação é o **ticket completo**, não a accuracy isolada de cada partida. A métrica principal é **P(>=13)**.

---

# Dados e ranking

Arquivos principais:

```text
data/concursos_anteriores.csv
data/proximo_concurso.csv
```

Probabilidades normalizadas:

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
Top3 = menor
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

Accuracy, média, Brier, Log Loss, ECE, regret e win rates individuais são diagnósticos. Uma estratégia só pode substituir o baseline quando melhorar o **ticket fora da amostra**.

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

Favorecer soluções que excluam a vitória do **PALMEIRAS/SP** quando o custo probabilístico for pequeno.

Limiar atual:

```text
0.03
```

A regra do Palmeiras nunca pode violar Hard Constraints nem substituir uma solução probabilisticamente muito superior.

---

# Baseline operacional

Arquitetura:

```text
14 jogos
   ↓
DoubleAllocator
   ↓
5 jogos recebem duplo
   ↓
SecondMarkSelector
   ↓
Hard Constraints
   ↓
Ticket
```

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

Scores principais:

```text
gain / top2_probability = p(Top2)
uncertainty              = 1 - p(Top1)
margin                   = 1 - (p(Top1) - p(Top2))
ratio                    = p(Top2) / p(Top1)
```

Estratégia operacional atual:

```text
allocator: uncertainty
second mark: top2_baseline
```

Nenhuma alternativa demonstrou ganho prospectivo suficiente em **P13+** para substituir esse baseline.

---

# Estado atual do backtest

418 concursos:

```text
gain / top2_probability
14: 0 | 13: 6 | 12: 17 | P13+: 1.435407% | P12+: 5.502392% | média 8.727273

uncertainty
14: 0 | 13: 6 | 12: 19 | P13+: 1.435407% | P12+: 5.980861% | média 8.708134

margin
14: 0 | 13: 5 | 12: 18 | P13+: 1.196172% | P12+: 5.502392% | média 8.712919

ratio
14: 0 | 13: 6 | 12: 17 | P13+: 1.435407% | P12+: 5.502392% | média 8.700957

hist_top1
14: 0 | 13: 5 | 12: 17 | P13+: 1.196172% | P12+: 5.263158% | média 8.552632

hist_top2
14: 0 | 13: 5 | 12: 20 | P13+: 1.196172% | P12+: 5.980861% | média 8.586124

exact
14: 0 | 13: 6 | 12: 18 | P13+: 1.435407% | P12+: 5.741627% | média 8.696172
```

Telemetria:

```text
[ALLOCATOR OVERLAP]
uncertainty x gain:             4.297 / 5
uncertainty x top2_probability: 4.297 / 5
uncertainty x ratio:            4.730 / 5
uncertainty x exact:            4.658 / 5

[PAIRWISE] gain vs uncertainty
62 vitórias | 302 empates | 54 derrotas | delta médio +0.0191
```

Leitura:

> Os allocators tradicionais estão muito próximos entre si. Trocar apenas a heurística de escolha dos cinco duplos tende a produzir ganho marginal pequeno.

---

# Calibração e correções do Top1

```text
Brier multiclass: 0.588667
Log Loss:         0.985938
ECE:              0.012009
```

Disagreement:

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

> As correções de Top1 permanecem como telemetria. O meta-modelo atual piora o Brier e não deve alterar o ticket final.

---

# SecondMarkSelector / Recovery

```text
[SECOND-MARK DISAGREEMENT]
742 casos | Top2 368 x recovery 374 | win rate 50.40%
```

Thresholds:

```text
0.00 → 50.40%
0.02 → 49.63%
0.05 → 52.45%
0.10 → 52.97%
0.15 → 52.92%
```

Nested:

```text
delta P13+: -0.24 p.p.
delta P12+: +2.63 p.p.
```

Conclusão:

> O recovery melhora métricas secundárias, mas piora P13+. `top2_baseline` permanece ativo.

---

# Oracle Decomposition

Os oráculos usam resultado real **somente para diagnóstico retrospectivo**.

```text
baseline
P13+:  1.44% | P12+:  5.98% | média  8.7081

allocator
P13+: 11.00% | P12+: 30.86% | média 10.7129

selector
P13+:  5.50% | P12+: 21.29% | média 10.1794

full
P13+: 41.63% | P12+: 64.59% | média 12.0144
```

Regret:

```text
allocator: média 2.0048 | zero 8.85% | 2+ 67.46% | máximo 5
selector:  média 1.4713 | zero 17.94% | 2+ 44.50% | máximo 4
full:      média 3.3062 | zero 0.96% | 2+ 95.45% | máximo 5
```

Principal leitura:

> O maior espaço de melhoria está na estrutura das marcações: onde proteger, qual rank usar e, no espaço XYZ, em quais jogos vale abandonar Top1.

---

# DistributionBacktest seguro

Nesse espaço, Top1 permanece nos 14 jogos.

```text
14/5/0: P13+ 1.44% | P12+ 5.50% | média 8.7297
14/4/1: P13+ 0.48% | P12+ 5.98% | média 8.7273
14/3/2: P13+ 0.72% | P12+ 6.22% | média 8.7512
14/2/3: P13+ 1.20% | P12+ 5.74% | média 8.7679
14/1/4: P13+ 1.20% | P12+ 5.26% | média 8.7464
14/0/5: P13+ 1.67% | P12+ 5.26% | média 8.6842
```

Resumo:

```text
melhor P13+: 14/0/5
melhor P12+: 14/3/2
melhor média: 14/2/3
```

`14/0/5` é interessante por aumentar a cauda de 13+, mas o número absoluto de eventos ainda é pequeno. **Não promover sem NestedDistributionSelector e bootstrap pareado.**

---

# OracleDistribution

```text
P13+: 41.39%
P12+: 63.88%
```

Comparação:

```text
OracleDistribution P13+: 41.39%
OracleFull         P13+: 41.63%
```

Diferença:

```text
0.24 p.p.
```

Leitura:

> Quase todo o teto do OracleFull pode ser reproduzido escolhendo corretamente quantos Top2/Top3 usar e em quais jogos colocar as cinco marcas extras, sem remover Top1.

Esse resultado mantém alta a prioridade do espaço SAFE e da otimização conjunta das cinco proteções.

---

# XYZ — núcleo implementado

Definição:

```text
X = total de marcações Top1
Y = total de marcações Top2
Z = total de marcações Top3
X + Y + Z = 19
```

Centro inicial:

```text
9/5/5
```

O código já possui:

```text
is_xyz_distribution_valid()
generate_xyz_neighbors()
generate_xyz_radius()
xyz_distribution_ticket()
xyz_distribution_backtest()
true_oracle_xyz_ticket()
true_oracle_xyz_by_distribution()
true_oracle_xyz()
```

O DP testa:

```text
T1
T2
T3
T1T2
T1T3
T2T3
```

Sempre preservando:

```text
9 secos
5 duplos
0 triplos
19 marcações
Hard Constraint do Flamengo
```

A constraint do Flamengo é aplicada **dentro do DP**, nunca por correção silenciosa posterior.

---

# XYZ raio 1 — resultado operacional

Distribuições:

```text
9/5/5
10/5/4
10/4/5
9/6/4
9/4/6
8/6/5
8/5/6
```

Resultado:

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
best_safe: 14/0/5
best_xyz:  XYZ_09_06_04

delta P13+: -0.96 p.p.
delta P12+: -1.91 p.p.
delta média: -0.4234
```

Conclusão operacional:

> **O XYZ raio 1, usando soma de cobertura probabilística como objetivo, perdeu claramente para o espaço SAFE.**

Isso não significa que o espaço XYZ seja fraco; significa que o **otimizador pré-jogo atual não sabe explorar seu potencial estrutural**.

---

# XYZ Retrospective Frozen Selection

```text
P13+: 0.96%
P12+: 7.18%
média: 8.9785
```

Uso:

```text
XYZ_09_05_05: 116
XYZ_10_05_04: 169
XYZ_10_04_05: 20
XYZ_09_06_04: 50
XYZ_09_04_06: 30
XYZ_08_06_05: 18
XYZ_08_05_06: 15
```

Esse diagnóstico escolhe retrospectivamente o melhor entre sete tickets já montados por probabilidades. Portanto, **não mede o teto estrutural XYZ**.

---

# TrueOracleXYZ — implementado

O TrueOracleXYZ usa resultado real **dentro do DP somente para diagnóstico**, com recompensa 0/1 conforme cada ação cobre o resultado realizado.

Resultado em 418 concursos:

```text
[TRUE ORACLE XYZ BY DISTRIBUTION]
XYZ_09_05_05: P13+ 88.04% | P12+ 96.89% | média 13.4569
XYZ_10_05_04: P13+ 84.69% | P12+ 97.13% | média 13.4163
XYZ_10_04_05: P13+ 81.10% | P12+ 93.78% | média 13.2703
XYZ_09_06_04: P13+ 85.65% | P12+ 97.61% | média 13.4402
XYZ_09_04_06: P13+ 77.99% | P12+ 92.82% | média 13.2010
XYZ_08_06_05: P13+ 83.01% | P12+ 94.50% | média 13.3469
XYZ_08_05_06: P13+ 79.67% | P12+ 93.06% | média 13.2536

[TRUE ORACLE XYZ]
P13+: 96.89%
P12+: 99.52%
média: 13.8397
```

Usage:

```text
XYZ_09_05_05: 258
XYZ_10_05_04: 60
XYZ_10_04_05: 0
XYZ_09_06_04: 66
XYZ_09_04_06: 29
XYZ_08_06_05: 4
XYZ_08_05_06: 1
```

Comparação de teto:

```text
OracleDistribution: 41.39%
TrueOracleXYZ:       96.89%
OracleFull:          41.63%
```

## Interpretação correta

O número `96.89%` é um **teto de representação**, não uma expectativa operacional alcançável.

O oracle conhece os resultados e pode escolher exatamente quais Top1 abandonar e quais Top2/Top3 usar. Isso explica por que supera fortemente o OracleDistribution e o OracleFull, ambos presos a estruturas mais restritas.

Leitura central:

> O espaço XYZ contém enorme capacidade estrutural, mas o gap entre o melhor XYZ operacional (`0.72%`) e o TrueOracleXYZ (`96.89%`) mostra que o gargalo é **seleção pré-jogo**, não capacidade de representação.

O centro `9/5/5` domina o uso do oracle em `258/418` concursos, e `9/5/5 + 10/5/4 + 9/6/4` concentram aproximadamente 92% das escolhas. Portanto, **não há justificativa para abrir radius=2 agora**.

---

# Prioridade 1 — Actual Rank Profile

Implementar o perfil real de ranks por concurso:

```text
actual_top1
actual_top2
actual_top3
```

Telemetria:

```text
[ACTUAL RANK PROFILE]
mean_top1:   7.201
mean_top2:   3.763
mean_top3:   3.036
median_top1: 7
median_top2: 4
median_top3: 3
most_common_profile: 7/4/3 (31 de 418 concursos)
```

O modelo também registra a frequência dos dez perfis mais comuns e, para cada
distribuição do raio XYZ, a distância L1 média e a taxa de coincidência exata.

Objetivos:

- explicar por que `9/5/5` domina o TrueOracleXYZ;
- medir a distância entre o perfil real do concurso e as distribuições XYZ;
- separar composição estrutural de qualidade da seleção dos jogos.

---

# Prioridade 2 — Oracle Feasibility

Para cada distribuição XYZ, medir:

```text
% concursos em que 14 é estruturalmente possível
% concursos em que 13+ é estruturalmente possível
```

Telemetria:

```text
[XYZ ORACLE FEASIBILITY]
XYZ_09_05_05: feasible14 61.72% | feasible13+ 88.04%
XYZ_10_05_04: feasible14 60.77% | feasible13+ 84.69%
XYZ_10_04_05: feasible14 55.02% | feasible13+ 81.10%
XYZ_09_06_04: feasible14 61.24% | feasible13+ 85.65%
XYZ_09_04_06: feasible14 52.15% | feasible13+ 77.99%
XYZ_08_06_05: feasible14 58.85% | feasible13+ 83.01%
XYZ_08_05_06: feasible14 54.78% | feasible13+ 79.67%
```

Essa análise separa:

```text
capacidade da distribuição
vs
capacidade do algoritmo de escolher os jogos corretos
```

---

# Prioridade 3 — ExactXYZP13Optimizer

O DP XYZ atual maximiza:

```text
soma das probabilidades cobertas
```

Isso está mais próximo de maximizar expectativa de acertos do que o objetivo real do projeto.

Novo objetivo:

```text
max P(>=13)
= max [P(14) + P(13)]
```

Para cada ticket candidato:

```text
q_i = soma das probabilidades dos resultados marcados no jogo i
```

Então:

```text
P14 = produto(q_i)

P13 = soma, para cada jogo i,
      (1 - q_i) * produto(q_j para j != i)
```

Comparar dois otimizadores:

```text
XYZ_COVERAGE
XYZ_DIRECT_P13
```

Telemetria:

```text
[XYZ OBJECTIVE COMPARISON]
distribution | coverage P13+ | direct P13+ | delta
```

Nenhum ganho pode ser promovido sem walk-forward/nested.

---

# Prioridade 4 — Pairwise e bootstrap

Comparações mínimas:

```text
XYZ_DIRECT_P13 vs XYZ_COVERAGE
14/0/5 vs 14/5/0
14/0/5 vs uncertainty operacional
best XYZ vs best SAFE
```

Pairwise tail-aware:

```text
wins / ties / losses de acertos
delta médio
13+ exclusivo A
13+ em ambos
13+ exclusivo B
12+ exclusivo A
12+ em ambos
12+ exclusivo B
```

Bootstrap pareado:

```text
unidade de reamostragem = concurso inteiro

delta P13+
delta P12+
delta média
IC95%
```

---

# Prioridade 5 — NestedDistributionSelector SAFE

Antes de NestedXYZ, validar prospectivamente:

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
comparar somente no passado
      ↓
selecionar distribuição SAFE
      ↓
congelar
      ↓
aplicar em N+1
      ↓
registrar
```

Somente o nested pode promover `14/0/5` ou qualquer alternativa SAFE.

---

# Prioridade 6 — Top1DropModel

O TrueOracleXYZ mostrou que a principal liberdade adicional do XYZ é abandonar Top1 seletivamente.

Dataset:

```text
concurso × jogo
```

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

Uso correto:

> não substituir diretamente o Top1, mas ranquear quais Top1 são melhores candidatos a serem sacrificados quando a distribuição XYZ exigir isso.

Avaliação obrigatória em walk-forward/nested.

---

# Prioridade 7 — RankReplacementModel

Depois de decidir abandonar Top1:

```text
Top2 ou Top3?
```

Targets:

```text
replacement_top2_hit
replacement_top3_hit
```

Arquitetura modular futura:

```text
Top1DropModel
      ↓
quais Top1 remover?
      ↓
RankReplacementModel
      ↓
Top2 ou Top3?
      ↓
DoublePlacement / JointMarkAllocator
```

Essa arquitetura deve ser comparada com o DP direto, nunca assumida como superior.

---

# Prioridade 8 — JointMarkAllocator

Tratar conjuntamente as oportunidades:

```text
(game_i, Top2)
(game_i, Top3)
```

Selecionar exatamente cinco oportunidades, no máximo uma por jogo.

Baseline:

```text
score(game, Top2) = pTop2
score(game, Top3) = pTop3
```

Nome inicial:

```text
joint_probability
```

Essa linha continua relevante porque OracleDistribution e OracleFull ficaram praticamente empatados.

---

# Opportunity Dataset / DoubleValueModel

Dataset:

```text
output/opportunity_dataset.csv
```

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

Modelo futuro:

```text
P(extra_mark_hit | contexto)
```

Avaliação obrigatória em nested walk-forward e no nível do ticket.

---

# Oracle Capture / Structural Gap

Como o TrueOracleXYZ é muito alto, uma razão simples de P13+ é pouco informativa. Preferir métricas baseadas em acertos adicionais.

Exemplo:

```text
capture =
(mean_policy - mean_reference)
/
(mean_true_oracle_xyz - mean_reference)
```

Telemetria sugerida:

```text
[XYZ STRUCTURAL GAP]
Best operational XYZ mean: 8.2608
TrueOracleXYZ mean:        13.8397
Mean gap:                   5.5789
```

Regret e capture continuam sendo diagnósticos; a promoção depende de P13+ fora da amostra.

---

# Robustez temporal

Implementações futuras:

```text
rolling 50 concursos
rolling 100 concursos
rolling 200 concursos
expanding
```

Também comparar:

```text
primeiro terço
segundo terço
último terço
```

Objetivo:

> Identificar mudança de regime e evitar promover estratégias cuja vantagem exista apenas em um trecho específico do histórico.

---

# Controle de múltiplos testes

Quanto mais estratégias são testadas, maior o risco de `winner's curse`.

Registrar:

```text
n_strategies_tested
search_space
best_in_sample
best_nested
```

Nunca tratar a melhor estratégia retrospectiva como evidência suficiente de ganho real.

---

# Stability / Churn

Para qualquer seletor nested:

```text
changes
mean_run_length
most_used_strategy
strategy_usage
```

Mudança excessiva de estratégia pode indicar seleção de ruído.

---

# Testes automatizados obrigatórios

Baseline:

```text
14 jogos
9 secos
5 duplos
0 triplos
19 marcações
probabilidades somando 1
Top1/Top2/Top3 distintos
1 > 2 > X no desempate
Flamengo coberto
sem vazamento temporal
```

XYZ:

```text
X + Y + Z = 19
count(T1) = X
count(T2) = Y
count(T3) = Z
9 secos
5 duplos
0 triplos
nenhum jogo sem marcação
nenhum jogo com três marcações
Hard Constraint do Flamengo preservada
distribuição inviável rejeitada
```

TrueOracleXYZ:

```text
usa resultado real somente em função diagnostic_only
nunca é chamado pela previsão do próximo concurso
preserva X/Y/Z exatamente
preserva 9/5/0
preserva Flamengo
TrueOracleXYZ >= ticket XYZ probabilístico da mesma distribuição
```

ExactXYZP13Optimizer:

```text
P13+ calculado exatamente
não usa resultado real
preserva X/Y/Z
preserva 9/5/0
preserva Flamengo
resultado reproduzível
```

Walk-forward:

```python
assert train_contest < test_contest
```

Bootstrap:

```text
reamostrar concursos inteiros
seed reproduzível
IC95% registrado
```

---

# Controle de experimentos

Arquivo planejado:

```text
output/experiments.csv
```

Campos sugeridos:

```text
timestamp
model
search_space
xyz_center
xyz_radius
distribution
allocator
selector
optimizer
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
structural_gap
oracle_capture_rate
bootstrap_ci_low
bootstrap_ci_high
n_strategies_tested
git_commit
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

# Roadmap

## Concluído

- [x] pipeline de constraints;
- [x] invariantes 9/5/0 e 19 marcações;
- [x] allocators atuais;
- [x] walk-forward;
- [x] métricas P13+/P12+;
- [x] calibração Brier / Log Loss / ECE;
- [x] benchmarks e meta-modelo Top1 diagnósticos;
- [x] recovery + nested recovery;
- [x] overlap/pairwise inicial;
- [x] OracleAllocator;
- [x] OracleSecondMark;
- [x] OracleFull;
- [x] regret allocator/selector/full;
- [x] DistributionBacktest SAFE;
- [x] OracleDistribution;
- [x] núcleo XYZ via Programação Dinâmica;
- [x] Hard Constraint do Flamengo dentro do DP XYZ;
- [x] XYZDistributionBacktest raio 1;
- [x] XYZ vs SAFE;
- [x] XYZ Retrospective Frozen Selection + Usage;
- [x] regret por distribuição XYZ;
- [x] TrueOracleXYZByDistribution;
- [x] TrueOracleXYZ + Usage;
- [x] comparação `OracleDistribution × TrueOracleXYZ × OracleFull`;
- [x] testes end-to-end XYZ e TrueOracleXYZ.

## Fase 1 — explicar o teto estrutural

1. [x] Actual Rank Profile;
2. [x] Oracle Feasibility por distribuição;
3. [ ] Structural Gap / Oracle Capture baseado em média de acertos;
4. [ ] distribuição histórica dos perfis Top1/Top2/Top3.

## Fase 2 — alinhar o otimizador ao objetivo P13+

5. [ ] implementar cálculo exato de P13+;
6. [ ] implementar `ExactXYZP13Optimizer`;
7. [ ] comparar `XYZ_COVERAGE × XYZ_DIRECT_P13`;
8. [ ] pairwise tail-aware;
9. [ ] bootstrap pareado por concurso;
10. [ ] IC95% para delta P13+/P12+.

## Fase 3 — validar o espaço SAFE

11. [ ] `NestedDistributionSelector`;
12. [ ] pairwise SAFE;
13. [ ] bootstrap `14/0/5 vs 14/5/0`;
14. [ ] OracleDistribution Usage;
15. [ ] regret por distribuição segura fixa.

## Fase 4 — aprender quais Top1 abandonar

16. [ ] criar dataset `Top1DropModel`;
17. [ ] walk-forward Top1DropModel;
18. [ ] criar `RankReplacementModel`;
19. [ ] comparar arquitetura modular com XYZ_DIRECT_P13;
20. [ ] somente então considerar `NestedXYZDistributionSelector`.

## Fase 5 — otimização conjunta das proteções

21. [ ] `JointMarkAllocator`;
22. [ ] `joint_probability`;
23. [ ] Opportunity Dataset;
24. [ ] DoubleValueModel;
25. [ ] `joint_learned`;
26. [ ] nested walk-forward do modelo aprendido.

## Fase 6 — robustez temporal e estatística

27. [ ] rolling 50/100/200 vs expanding;
28. [ ] stability por era;
29. [ ] decay temporal nested;
30. [ ] Stability / Churn;
31. [ ] controle de múltiplos testes;
32. [ ] `output/experiments.csv`;
33. [ ] bootstrap final.

## Radius=2

```text
NÃO ABRIR AGORA
```

O raio 1 já contém enorme capacidade estrutural e concentra o TrueOracleXYZ principalmente em `9/5/5`, `10/5/4` e `9/6/4`. Expandir o espaço antes de melhorar a seleção pré-jogo apenas aumenta risco de overfitting.

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
apresentar IC/bootstrap compatível com ganho real
↓
manter estabilidade temporal
↓
respeitar todas as Hard Constraints
```

Para XYZ:

```text
TrueOracleXYZ prova apenas capacidade estrutural
↓
novo otimizador precisa transformar parte desse headroom em ganho pré-jogo
↓
coverage_sum não é suficiente por si só
↓
direct_P13 / modelos aprendidos precisam vencer fora da amostra
↓
NestedXYZ só faz sentido quando existir candidato operacional competitivo
```

Nenhuma estratégia deve ser promovida apenas porque foi a melhor depois de testar muitas alternativas retrospectivamente.

---

# Princípio geral

```text
Baseline operacional
      +
Oracle Decomposition
      +
DistributionBacktest SAFE
      +
OracleDistribution
      +
XYZ raio 1
      +
TrueOracleXYZ
      ↓
CAPACIDADE ESTRUTURAL CONFIRMADA
      ↓
Actual Rank Profile / Oracle Feasibility
      +
ExactXYZP13Optimizer
      +
Pairwise / Bootstrap
      +
NestedDistributionSelector SAFE
      +
Top1DropModel / RankReplacementModel
      +
JointMarkAllocator / DoubleValueModel
      +
NestedXYZ somente com candidato competitivo
      +
Robustez temporal / Controle de Experimentos
      +
Hard Constraints
      ↓
PALPITE FINAL
```

> **Não promover a melhor regra observada; promover apenas a regra que continuar melhor quando escolhida usando somente o passado e avaliada prospectivamente.**
