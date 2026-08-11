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

Princípio metodológico:

> Melhorar uma métrica intermediária não significa melhorar o objetivo final do ticket.

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

> O maior espaço de melhoria está na **estrutura das cinco marcações extras**: onde colocá-las e qual rank usar como proteção.

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

> Quase todo o teto do OracleFull pode ser reproduzido escolhendo corretamente quantos Top2/Top3 usar e em quais jogos colocar as cinco marcas extras, **sem remover Top1 dos 14 jogos**.

Esse resultado aumenta a prioridade do espaço SAFE e da otimização conjunta das cinco proteções.

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

# XYZ raio 1 — resultado atual

Distribuições testadas:

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

> **O XYZ raio 1, usando a função objetivo atual de soma de cobertura probabilística, perdeu claramente para o espaço SAFE.**

XYZ não deve ser promovido nem expandido para raio 2 neste momento.

---

# XYZ Retrospective Frozen Selection

A execução atual mostra:

```text
[ORACLE XYZ]
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

Porém, metodologicamente esse número **não é comparável diretamente** ao OracleDistribution ou OracleFull.

O fluxo atual é:

```text
probabilidades pré-jogo
      ↓
gerar um ticket otimizado para cada XYZ
      ↓
congelar esses tickets
      ↓
usar o resultado real apenas para escolher retrospectivamente
qual dos sete tickets congelados acertou mais
```

Essa telemetria agora é exposta pelo nome conceitualmente correto:

```text
XYZ Retrospective Frozen Selection
```

ou:

```text
XYZ Best Frozen Ticket
```

O número `0.96%` mede o benefício de escolher retrospectivamente entre sete tickets já congelados — **não o teto estrutural do espaço XYZ**. O código e a saída não usam mais o rótulo ambíguo `OracleXYZ` para essa seleção.

---

# TrueOracleXYZ — implementado

O verdadeiro oracle estrutural XYZ foi implementado como diagnóstico isolado da previsão.

Para cada jogo, usando o resultado real apenas no diagnóstico:

```text
T1   = 1 se Top1 contém o resultado real
T2   = 1 se Top2 contém o resultado real
T3   = 1 se Top3 contém o resultado real
T1T2 = 1 se o resultado está em {Top1, Top2}
T1T3 = 1 se o resultado está em {Top1, Top3}
T2T3 = 1 se o resultado está em {Top2, Top3}
```

Objetivo do DP retrospectivo:

```text
maximizar número de acertos
```

sujeito a:

```text
X Top1
Y Top2
Z Top3
9 secos
5 duplos
19 marcações
Hard Constraint do Flamengo
```

Funções sugeridas:

```text
true_oracle_xyz_ticket(...)
true_oracle_xyz_by_distribution(...)
true_oracle_xyz(...)
```

Telemetria em 418 concursos:

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
mean: 13.8397
usage: 258 / 60 / 0 / 66 / 29 / 4 / 1
```

Comparar:

```text
OracleDistribution
TrueOracleXYZ
OracleFull
```

Leitura:

> O espaço XYZ possui headroom estrutural muito superior ao obtido pelo otimizador probabilístico atual. O fracasso operacional vem da escolha das marcações dentro do espaço, e não da ausência de tickets XYZ capazes de atingir a cauda 13+.

Regra de decisão:

```text
TrueOracleXYZ > OracleDistribution
    → existe headroom estrutural e vale investigar um otimizador XYZ melhor
```

O resultado é apenas um teto retrospectivo: usa os resultados reais dentro do DP e **nunca pode selecionar o palpite do próximo concurso**.

---

# Prioridade 2 — NestedDistributionSelector SAFE

Antes de construir um NestedXYZ, validar prospectivamente as seis distribuições seguras:

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
comparar apenas no passado
      ↓
selecionar distribuição SAFE
      ↓
congelar
      ↓
aplicar em N+1
      ↓
registrar
```

Somente o nested pode promover `14/0/5` ou qualquer outra distribuição segura.

---

# Prioridade 3 — Pairwise e bootstrap SAFE

Comparações mínimas:

```text
14/0/5 vs 14/5/0
14/0/5 vs uncertainty operacional
14/3/2 vs 14/5/0
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

Com poucos eventos 13+, diferenças como `1.67% vs 1.44%` podem representar apenas um concurso.

---

# Prioridade 4 — ExactXYZP13Optimizer

Somente se o TrueOracleXYZ mostrar headroom estrutural relevante.

O DP XYZ atual maximiza:

```text
soma das probabilidades cobertas
```

Isso está mais próximo de maximizar expectativa de acertos do que a função objetivo real.

Novo objetivo:

```text
max P(>=13)
= max [P(14) + P(13)]
```

Para um ticket candidato:

```text
q_i = soma das probabilidades dos resultados marcados no jogo i
```

A distribuição de acertos pode ser calculada exatamente pela Poisson-binomial.

Comparar:

```text
coverage_sum
vs
direct_P13_optimizer
```

Nenhum ganho pode ser promovido sem nested walk-forward.

---

# Prioridade 5 — JointMarkAllocator

A arquitetura atual separa:

```text
qual jogo recebe duplo?
+
qual segunda marca recebe?
```

Mas a decisão real pode ser tratada conjuntamente.

Oportunidades:

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

Essa linha ganha prioridade porque o OracleDistribution está praticamente empatado com o OracleFull sem precisar remover Top1.

---

# Opportunity Dataset / DoubleValueModel

Dataset futuro:

```text
output/opportunity_dataset.csv
```

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
posição
perfil do concurso
```

Target principal:

```text
extra_mark_hit
```

onde:

```text
extra_mark_hit = 1
```

se aquela marca adicional recuperaria um erro do Top1.

Modelo futuro:

```text
P(extra_mark_hit | contexto)
```

Avaliação obrigatória em nested walk-forward e no nível do ticket.

---

# Oracle Capture Rate

Para medir quanto potencial uma arquitetura consegue extrair:

```text
oracle_capture_rate =
    (hits_policy - hits_baseline)
    /
    (hits_oracle - hits_baseline)
```

Para XYZ, usar **TrueOracleXYZ**, nunca o Frozen Selection atual.

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
- [x] DistributionBacktest seguro `14/5/0 → 14/0/5`;
- [x] OracleDistribution;
- [x] núcleo XYZ via Programação Dinâmica;
- [x] Hard Constraint do Flamengo dentro do DP XYZ;
- [x] XYZDistributionBacktest raio 1;
- [x] XYZ vs SAFE;
- [x] Frozen Selection XYZ + Usage;
- [x] regret por distribuição XYZ;
- [x] testes end-to-end XYZ.

## Fase 1 — diagnóstico estrutural decisivo

1. [x] renomear o OracleXYZ atual para `XYZ Retrospective Frozen Selection`;
2. [x] implementar `true_oracle_xyz_ticket()`;
3. [x] implementar `TrueOracleXYZByDistribution`;
4. [x] implementar `TrueOracleXYZ` + Usage;
5. [x] comparar `OracleDistribution × TrueOracleXYZ × OracleFull`;
6. [ ] calcular Oracle Capture Rate XYZ.

## Fase 2 — validar o espaço SAFE

7. [ ] implementar `NestedDistributionSelector`;
8. [ ] pairwise SAFE tail-aware;
9. [ ] bootstrap pareado por concurso;
10. [ ] IC95% para delta P13+/P12+;
11. [ ] OracleDistribution Usage;
12. [ ] regret por distribuição segura fixa.

## Fase 3 — decisão sobre continuidade XYZ

13. [ ] decidir se XYZ possui headroom estrutural superior ao SAFE;
14. [ ] somente se sim, implementar `ExactXYZP13Optimizer`;
15. [ ] comparar `coverage_sum × direct_P13_optimizer`;
16. [ ] somente depois considerar `NestedXYZDistributionSelector`;
17. [ ] radius=2 apenas se houver evidência estrutural + nested.

## Fase 4 — otimização conjunta das marcas extras

18. [ ] `JointMarkAllocator`;
19. [ ] `joint_probability`;
20. [ ] Top1-only baseline;
21. [ ] Extra Mark Efficiency;
22. [ ] Oracle Capture Rate SAFE/Joint;
23. [ ] Recovery Profile.

## Fase 5 — aprendizado de valor da marca extra

24. [ ] Opportunity Dataset;
25. [ ] DoubleValueModel;
26. [ ] `joint_learned`;
27. [ ] nested walk-forward do modelo aprendido.

## Fase 6 — robustez temporal e estatística

28. [ ] rolling 50/100/200 vs expanding;
29. [ ] stability por era;
30. [ ] decay temporal nested;
31. [ ] Stability / Churn;
32. [ ] controle de múltiplos testes;
33. [ ] `output/experiments.csv`;
34. [ ] bootstrap final.

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
TrueOracleXYZ deve justificar a continuação da linha
↓
o ganho deve existir além do Frozen Selection retrospectivo
↓
se houver novo otimizador, ele deve vencer em nested
↓
remover Top1 precisa produzir benefício mensurável
```

Nenhuma estratégia deve ser promovida apenas porque foi a melhor depois de testar muitas alternativas retrospectivamente.

---

# Princípio geral

```text
Baseline seguro
      +
Oracle Decomposition
      +
DistributionBacktest SAFE
      +
OracleDistribution
      +
XYZ raio 1 já testado
      +
TrueOracleXYZ
      ↓
DECISÃO: continuar ou congelar XYZ
      ↓
NestedDistributionSelector SAFE
      +
Pairwise / Bootstrap
      +
JointMarkAllocator
      +
Opportunity Dataset / DoubleValueModel
      +
Exact P13+ optimizer somente onde houver headroom
      +
Robustez temporal / Controle de Experimentos
      +
Hard Constraints
      ↓
PALPITE FINAL
```

> **Não promover a melhor regra observada; promover apenas a regra que continuar melhor quando escolhida usando somente o passado e avaliada prospectivamente.**
