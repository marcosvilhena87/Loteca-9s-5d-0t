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

Accuracy, média, Brier, Log Loss, ECE e win rates individuais são diagnósticos. Uma estratégia só pode substituir o baseline quando melhorar o **ticket fora da amostra**.

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

# Baseline atual

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

Políticas atuais de allocator:

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

`gain` e `top2_probability` são equivalentes no baseline atual.

A estratégia operacional permanece:

```text
allocator: uncertainty
second mark: top2_baseline
```

Isso não significa dominância estatística robusta; significa apenas que nenhuma alternativa demonstrou ganho prospectivo suficiente em **P13+** para substituir o baseline.

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

# Calibração e qualidade probabilística

Resultado atual:

```text
Brier multiclass: 0.588667
Log Loss:         0.985938
ECE:              0.012009
```

Essas métricas continuam sendo diagnósticas. Uma melhoria de calibração só interessa operacionalmente se melhorar **P13+ do ticket fora da amostra**.

---

# Correções do Top1

Resultados atuais de disagreement:

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

> As correções de Top1 permanecem como benchmarks/telemetria e não alteram o ticket final. O meta-modelo atual piora o Brier e perde para o baseline nos disagreements informativos.

---

# SecondMarkSelector / Recovery

Resultado atual:

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

> O recovery melhora métricas secundárias, especialmente P12+, mas piora P13+. `top2_baseline` permanece ativo.

Esse resultado é um princípio metodológico importante do projeto:

> Melhorar uma métrica intermediária não significa melhorar o objetivo final do ticket.

---

# Oracle Decomposition — implementado

Os oráculos usam resultado real **somente para diagnóstico retrospectivo**.

```text
[ORACLE DECOMPOSITION]

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
[REGRET ALLOCATOR]
média 2.0048 | zero 8.85% | 2+ 67.46% | máximo 5

[REGRET SELECTOR]
média 1.4713 | zero 17.94% | 2+ 44.50% | máximo 4

[REGRET FULL]
média 3.3062 | zero 0.96% | 2+ 95.45% | máximo 5
```

Principal leitura:

> O maior espaço de melhoria está na **estrutura das cinco marcações extras**: onde colocá-las e qual rank usar como proteção.

O OracleAllocator sozinho leva P13+ de aproximadamente `1.44%` para `11.00%`, evidenciando que a alocação das proteções é um gargalo estrutural.

---

# DistributionBacktest seguro — implementado

Nesse espaço, Top1 permanece nos 14 jogos e as cinco marcas extras são distribuídas entre Top2 e Top3.

Distribuições:

```text
14/5/0
14/4/1
14/3/2
14/2/3
14/1/4
14/0/5
```

Resultado atual:

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

`14/0/5` é interessante por aumentar a cauda de P13+, mas o número absoluto de eventos 13+ ainda é pequeno. A diferença não justifica promoção sem validação nested e bootstrap pareado.

---

# OracleDistribution — implementado

Resultado atual:

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

Leitura principal:

> Quase todo o teto do OracleFull pode ser reproduzido escolhendo corretamente **quantos Top2/Top3 usar e em quais jogos colocar as cinco marcas extras**.

Essa é a principal justificativa para a linha de pesquisa XYZ.

---

# NestedDistributionSelector — pendente

Fluxo obrigatório:

```text
histórico até N
      ↓
comparar distribuições somente no passado
      ↓
selecionar
      ↓
congelar
      ↓
aplicar no concurso N+1
      ↓
registrar
      ↓
repetir
```

Somente o resultado nested pode promover uma distribuição segura diferente do baseline atual.

---

# XYZ — estado atual

## O núcleo XYZ já está implementado

O código atual já possui:

```text
is_xyz_distribution_valid()
generate_xyz_neighbors()
generate_xyz_radius()
xyz_distribution_ticket()
```

O `xyz_distribution_ticket()` usa **Programação Dinâmica** para testar, por jogo:

```text
T1
T2
T3
T1T2
T1T3
T2T3
```

Estado conceitual:

```text
(used_T1, used_T2, used_T3, used_doubles)
```

Objetivo final:

```text
used_T1 = X
used_T2 = Y
used_T3 = Z
used_doubles = 5
```

A Hard Constraint do Flamengo é aplicada **durante a geração dos estados**, e não por correção posterior. Assim, uma distribuição XYZ inviável é rejeitada em vez de ser silenciosamente alterada.

Também já existem testes automatizados para:

```text
vizinhança única e válida
transferência unitária entre X/Y/Z
preservação exata de X/Y/Z
9 secos / 5 duplos / 19 marcações
Hard Constraint do Flamengo
rejeição de distribuição inviável
```

## Integração end-to-end

O núcleo XYZ está integrado à execução principal de:

```powershell
python main.py
```

`main.py` agora registra:

```text
XYZDistributionBacktest (raio 1)
OracleXYZ e Usage
regret por distribuição
XYZ vs SAFE
telemetria completa em main.py
```

Nos 418 concursos fora da amostra, o melhor candidato fixo do raio 1 foi
`9/6/4` (`P13+ 0.72%`, `P12+ 3.35%`, média `8.2608`). O benchmark seguro
`14/0/5` permaneceu superior, com deltas XYZ de `-0.96 p.p.` em P13+,
`-1.91 p.p.` em P12+ e `-0.4234` acerto médio. O OracleXYZ limitado às sete
distribuições atingiu `P13+ 0.96%`, `P12+ 7.18%` e média `8.9785`.

Portanto, XYZ **não foi promovido** ao ticket operacional. Ainda faltam pairwise,
bootstrap e seleção nested antes de qualquer reconsideração prospectiva.

---

# Definição XYZ

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

Diferentemente do espaço seguro `14/x/y`, o espaço XYZ pode remover Top1 de alguns jogos.

Sempre:

```text
9 secos
5 duplos
0 triplos
19 marcações
```

---

# Vizinhança ±1 de 9/5/5

Raio 0:

```text
9/5/5
```

Raio 1:

```text
10/4/5
10/5/4
8/6/5
8/5/6
9/6/4
9/4/6
```

Cada vizinho resulta de uma única transferência de uma marcação entre duas colunas.

Gerador:

```text
(X+1,Y-1,Z)
(X+1,Y,Z-1)
(X-1,Y+1,Z)
(X,Y+1,Z-1)
(X-1,Y,Z+1)
(X,Y-1,Z+1)
```

Regras:

```text
X >= 0
Y >= 0
Z >= 0
X + Y + Z = 19
sem duplicatas
viável com 9 secos / 5 duplos
```

---

# Prioridade 1 — XYZDistributionBacktest

Primeiro experimento XYZ end-to-end:

```text
9/5/5
10/4/5
10/5/4
8/6/5
8/5/6
9/6/4
9/4/6
```

Telemetria esperada:

```text
[XYZ DISTRIBUTION BACKTEST]

X/Y/Z      14   13   12    P13+     P12+     mean
9/5/5       ...  ...  ...     ...       ...      ...
10/4/5      ...  ...  ...     ...       ...      ...
10/5/4      ...  ...  ...     ...       ...      ...
8/6/5       ...  ...  ...     ...       ...      ...
8/5/6       ...  ...  ...     ...       ...      ...
9/6/4       ...  ...  ...     ...       ...      ...
9/4/6       ...  ...  ...     ...       ...      ...
```

Adicionar também:

```text
median
stddev
regret
```

Pergunta principal:

> Remover alguns Top1 e redistribuir as 19 marcações melhora P13+ fora da amostra?

---

# Prioridade 2 — XYZ vs SAFE

Comparar o melhor XYZ com os benchmarks seguros no mesmo conjunto de concursos.

```text
[XYZ VS SAFE]
best_safe:
best_xyz:
delta P13+:
delta P12+:
delta mean:
```

Benchmarks mínimos:

```text
14/5/0
14/0/5
9/5/5
```

---

# Prioridade 3 — OracleXYZ

Para cada concurso, usando o resultado real apenas como diagnóstico retrospectivo:

```text
qual distribuição XYZ permitida teria produzido mais acertos?
```

Telemetria:

```text
[ORACLE XYZ]
P13+:
P12+:
mean:

[ORACLE XYZ USAGE]
9/5/5: ...
10/4/5: ...
10/5/4: ...
8/6/5: ...
8/5/6: ...
9/6/4: ...
9/4/6: ...
```

Comparar obrigatoriamente com:

```text
OracleDistribution P13+: 41.39%
OracleFull P13+:         41.63%
```

Objetivo:

> Medir quanto do teto estrutural é capturado pelo espaço XYZ de raio controlado.

Se `OracleXYZ` ficar muito abaixo do OracleDistribution/OracleFull, ampliar o espaço XYZ terá baixo valor marginal. Se ficar próximo, a seleção de distribuição torna-se um alvo central do projeto.

---

# Regret por distribuição XYZ

Para cada distribuição fixa, registrar:

```text
regret médio
regret mediano
regret = 0
regret = 1
regret >= 2
regret máximo
```

Isso permite identificar distribuições robustas que permanecem próximas do oracle mesmo quando não lideram P13+ bruto.

---

# Pairwise e bootstrap

## Pairwise contest-by-contest

Comparar cada candidato com benchmarks seguros:

```text
wins
ties
losses
delta médio de acertos
P13+ wins/ties/losses
P12+ wins/ties/losses
```

## Bootstrap pareado

A unidade de reamostragem deve ser o **concurso inteiro**, nunca partidas individuais.

Registrar:

```text
delta P13+
delta P12+
delta mean
IC95%
```

Motivo:

> Com aproximadamente 418 concursos e poucos eventos 13+, diferenças aparentemente relevantes podem representar apenas um ou dois concursos.

---

# NestedXYZDistributionSelector

Fluxo:

```text
histórico até N
      ↓
gerar espaço XYZ permitido
      ↓
comparar apenas no passado
      ↓
selecionar X/Y/Z
      ↓
congelar
      ↓
aplicar no N+1
      ↓
registrar
```

Somente o nested pode promover XYZ para o ticket final.

Critério sugerido de seleção:

```text
1. P13+
2. número de 14
3. número de 13
4. P12+
5. média
6. menor variância
7. menor distância de 9/5/5
```

Empates devem ser resolvidos de forma conservadora para reduzir seleção por ruído.

---

# Regularização do seletor XYZ

Como P13+ é um evento raro, não usar frequência bruta de maneira ingênua para selecionar distribuições.

Experimentar:

```text
Beta-Binomial
shrinkage em direção ao baseline
mínimo de evidência
penalização por complexidade/distância do centro
```

Objetivo:

> Evitar que um único concurso de 13 pontos provoque mudança artificial de distribuição.

---

# Stability / Churn

Telemetria futura:

```text
[XYZ SELECTION STABILITY]
changes:
mean_run_length:
most_used_distribution:
distribution_usage:
```

Se o seletor mudar de distribuição quase todo concurso, isso pode indicar ruído. Regimes persistentes são mais interessantes e mais plausíveis de generalizar.

---

# Busca em raios

Não abrir todo o espaço XYZ de uma vez.

```text
raio 0 → 9/5/5
raio 1 → seis vizinhos unitários
raio 2 → somente se raio 1 mostrar sinal nested
raio 3+ → somente com evidência adicional
```

Isso reduz `winner's curse` e overfitting por busca excessiva.

---

# Próxima fronteira — otimizar diretamente P(>=13)

O DP XYZ atual maximiza a **soma das probabilidades cobertas** pelas escolhas.

Isso é um bom baseline, mas está mais próximo de maximizar expectativa de acertos do que de maximizar diretamente P13+.

Dois tickets podem apresentar:

```text
Ticket A
E[acertos] = 9.1
P13+       = 0.35%

Ticket B
E[acertos] = 8.9
P13+       = 0.48%
```

Para este projeto, o Ticket B é superior.

Implementação futura sugerida:

```text
ExactXYZP13Optimizer
```

Objetivo:

```text
max P(>=13)
= max [P(14) + P(13)]
```

A probabilidade do ticket pode ser calculada pela distribuição Poisson-binomial das probabilidades de acerto dos 14 jogos.

Essa etapa deve ser realizada **depois** da validação estrutural XYZ, para evitar aumentar simultaneamente o espaço de busca e a complexidade do otimizador.

---

# Top1-only e valor das cinco marcas extras

Implementações futuras:

```text
Top1-only baseline
Extra Mark Efficiency
Oracle Capture Rate
Recovery Profile
```

Fórmulas:

```text
extra_mark_efficiency = (hits_ticket - hits_top1_only) / 5
```

```text
oracle_capture_rate =
    (hits_policy - hits_top1_only)
    /
    (hits_oracle_full - hits_top1_only)
```

Essas métricas ajudam a medir quanto valor cada arquitetura extrai das cinco marcações adicionais.

---

# JointMarkAllocator

Experimento paralelo à arquitetura `DoubleAllocator → SecondMarkSelector`.

Para cada jogo:

```text
T1T2 → score_T2
T1T3 → score_T3
```

Baseline inicial:

```text
score_T2 = pTop2
score_T3 = pTop3
```

Selecionar exatamente cinco oportunidades, no máximo uma por jogo.

Nome inicial:

```text
joint_probability
```

Essa linha deve ficar atrás de XYZ no roadmap enquanto o espaço estrutural ainda não estiver totalmente diagnosticado.

---

# Opportunity Dataset / DoubleValueModel

Dataset futuro:

```text
output/opportunity_dataset.csv
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

Targets:

```text
extra_gain_top2
extra_gain_top3
```

Saídas futuras:

```text
score_T2 ≈ P(extra_gain_top2 = 1 | contexto)
score_T3 ≈ P(extra_gain_top3 = 1 | contexto)
```

Qualquer modelo aprendido deve ser avaliado em nested walk-forward e comparado no nível do ticket.

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

Walk-forward:

```python
assert train_contest < test_contest
```

Testes futuros obrigatórios:

```text
XYZDistributionBacktest sem leakage
OracleXYZ marcado como diagnostic_only
nested seleciona somente com dados passados
bootstrap reamostra concursos inteiros
requested_distribution == effective_distribution em XYZ válido
P13 optimizer nunca viola constraints
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
- [x] benchmarks Top1;
- [x] Top1-meta diagnóstico;
- [x] recovery + nested recovery;
- [x] overlap/pairwise inicial;
- [x] OracleAllocator;
- [x] OracleSecondMark;
- [x] OracleFull;
- [x] regret allocator/selector/full;
- [x] DistributionBacktest seguro `14/5/0 → 14/0/5`;
- [x] OracleDistribution;
- [x] `is_xyz_distribution_valid()`;
- [x] `generate_xyz_neighbors()`;
- [x] `generate_xyz_radius()`;
- [x] `xyz_distribution_ticket()` via Programação Dinâmica;
- [x] Hard Constraint do Flamengo integrada ao DP XYZ;
- [x] testes unitários do núcleo XYZ.

## Fase 1 — XYZ end-to-end

1. [x] implementar `XYZDistributionBacktest` para `9/5/5 ± raio 1`;
2. [x] imprimir `[XYZ DISTRIBUTION BACKTEST]` em `main.py`;
3. [x] comparar `best_safe × best_xyz`;
4. [x] implementar `OracleXYZ`;
5. [x] imprimir `OracleXYZ Usage`;
6. [x] implementar regret por distribuição XYZ;
7. [x] adicionar testes end-to-end XYZ sem leakage.

## Fase 2 — significância e robustez

8. [ ] pairwise XYZ vs safe;
9. [ ] bootstrap pareado por concurso;
10. [ ] IC95% para delta P13+/P12+;
11. [ ] `OracleDistribution Usage`;
12. [ ] regret por distribuição segura fixa;
13. [ ] `NestedDistributionSelector` seguro.

## Fase 3 — validação prospectiva XYZ

14. [ ] `NestedXYZDistributionSelector` raio 1;
15. [ ] regularização/shrinkage do seletor;
16. [ ] Stability / Churn;
17. [ ] decidir se raio 2 merece ser aberto;
18. [ ] promover XYZ somente se superar baseline seguro em nested.

## Fase 4 — objetivo direto P13+

19. [ ] implementar cálculo exato de P13+ para ticket candidato;
20. [ ] criar `ExactXYZP13Optimizer`;
21. [ ] comparar `coverage_sum × direct_P13_optimizer`;
22. [ ] validar em nested walk-forward.

## Fase 5 — valor das marcações extras

23. [ ] Top1-only;
24. [ ] Extra Mark Efficiency;
25. [ ] Oracle Capture Rate;
26. [ ] Recovery Profile.

## Fase 6 — otimização conjunta

27. [ ] `JointMarkAllocator`;
28. [ ] `joint_probability`;
29. [ ] Opportunity Dataset;
30. [ ] DoubleValueModel;
31. [ ] `joint_learned`.

## Fase 7 — robustez final

32. [ ] rolling vs expanding;
33. [ ] decay temporal nested;
34. [ ] reliability tables;
35. [ ] `output/experiments.csv`;
36. [ ] bootstrap final;
37. [ ] relatório comparativo de todas as estratégias promovíveis.

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

Para XYZ, acrescentar:

```text
superar o baseline seguro em nested
↓
justificar a remoção de Top1 em alguns jogos
↓
mostrar que o ganho não depende de uma única distribuição extrema
↓
mostrar estabilidade suficiente da distribuição selecionada
```

Nenhuma estratégia deve ser promovida apenas porque foi a melhor depois de testar muitas alternativas retrospectivamente.

---

# Princípio geral

```text
Baseline seguro
      +
Oracle Decomposition
      +
DistributionBacktest seguro
      +
XYZ DP já implementado
      +
XYZDistributionBacktest
      +
OracleXYZ
      +
Pairwise / Bootstrap
      +
NestedDistributionSelector
      +
NestedXYZDistributionSelector
      +
ExactXYZP13Optimizer
      +
JointMarkAllocator / DoubleValueModel
      +
Regret / Stability / Controle de Experimentos
      +
Hard Constraints
      ↓
PALPITE FINAL
```

> **Não promover a melhor regra observada; promover apenas a regra que continuar melhor quando escolhida usando somente o passado e avaliada prospectivamente.**
