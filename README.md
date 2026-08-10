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
445 concursos
30 concursos na janela inicial
415 concursos avaliados em walk-forward

Top1: 51.7817%
Top2: 26.5329%
Top3: 21.6854%
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

---

# Estado atual do backtest

415 concursos:

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

`uncertainty` permanece como escolha operacional por desempate, não por dominância estatística robusta.

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

---

# Correções do Top1

Resultados atuais:

```text
top1_residual:    48.32% de win rate histórico
top1_lift:        48.52%
top1_reliability: 47.61%
p(top1_meta):     44.01%
```

Top1-meta:

```text
Brier baseline: 0.233977
Brier meta:     0.240629
```

Conclusão:

> As correções de Top1 permanecem como benchmarks/telemetria e não alteram o ticket final.

---

# SecondMarkSelector / Recovery

```text
[SECOND-MARK DISAGREEMENT]
739 casos | Top2 368 x recovery 371 | win rate 50.20%
```

Nested:

```text
Top2 baseline:   P13+ 1.4458% | P12+ 6.0241% | média 8.7205
Nested recovery: P13+ 1.2048% | P12+ 8.6747% | média 8.7759

delta P13+: -0.2410 p.p.
delta P12+: +2.6506 p.p.
```

Conclusão:

> O recovery melhora P12+ e média, mas piora P13+. `top2_baseline` permanece ativo.

---

# Oracle Decomposition — implementado

Os oráculos usam resultado real **somente para diagnóstico retrospectivo**.

```text
[ORACLE DECOMPOSITION]

baseline
P13+:  1.45% | P12+:  6.02% | média  8.7205

allocator
P13+: 11.08% | P12+: 31.08% | média 10.7229

selector
P13+:  5.54% | P12+: 21.45% | média 10.1831

full
P13+: 41.93% | P12+: 65.06% | média 12.0289
```

Regret:

```text
[REGRET ALLOCATOR]
média 2.0024 | zero 8.92% | 2+ 67.23% | máximo 5

[REGRET SELECTOR]
média 1.4627 | zero 18.07% | 2+ 44.34% | máximo 4

[REGRET FULL]
média 3.3084 | zero 0.96% | 2+ 95.42% | máximo 5
```

Principal leitura:

> Há muito espaço estrutural para melhorar **onde colocar as cinco marcações extras** e **qual rank usar como proteção**.

---

# DistributionBacktest seguro — implementado

Nesse espaço Top1 permanece nos 14 jogos.

Distribuições testadas:

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
14/5/0: P13+ 1.45% | P12+ 5.54% | média 8.7446
14/4/1: P13+ 0.48% | P12+ 6.02% | média 8.7446
14/3/2: P13+ 0.72% | P12+ 6.27% | média 8.7639
14/2/3: P13+ 1.20% | P12+ 5.78% | média 8.7807
14/1/4: P13+ 1.20% | P12+ 5.30% | média 8.7590
14/0/5: P13+ 1.69% | P12+ 5.30% | média 8.6940
```

Resumo:

```text
melhor P13+: 14/0/5
melhor P12+: 14/3/2
melhor média: 14/2/3
```

A diferença de P13+ entre `14/0/5` e `14/5/0` é pequena e não justifica promoção sem nested/bootstrap.

---

# OracleDistribution — implementado

```text
P13+: 41.69%
P12+: 64.34%
```

Comparação:

```text
OracleDistribution P13+: 41.69%
OracleFull         P13+: 41.93%
```

A diferença de apenas `0.24 p.p.` sugere que quase todo o teto do espaço seguro pode ser explicado por:

```text
quantos Top2/Top3 usar
+
em quais jogos colocá-los
```

Próximas telemetrias:

```text
OracleDistribution usage
regret por distribuição fixa
pairwise P13+/P12+
bootstrap pareado
requested_distribution vs effective_distribution
```

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

Somente o resultado nested pode promover uma distribuição diferente do baseline seguro.

---

# Nova prioridade — XYZDistributionBacktest

## Status real

O README já descrevia a arquitetura XYZ, porém a execução atual de:

```powershell
python main.py
```

**ainda não mostra nenhum bloco `[XYZ ...]`**.

Portanto, a linha XYZ deve ser considerada **não integrada end-to-end ao pipeline principal**. Helpers ou protótipos eventualmente presentes no código não contam como concluídos enquanto não houver:

```text
backtest XYZ executável
telemetria no terminal
testes automatizados
OracleXYZ
integração em main.py
```

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

Marcas permitidas:

```text
SECO
T1
T2
T3

DUPLO
T1T2
T1T3
T2T3
```

Sempre:

```text
9 secos
5 duplos
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

Cada vizinho deve resultar de **uma única transferência de 1 marcação entre duas colunas**.

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

# Implementação recomendada — XYZ por Programação Dinâmica

Para cada jogo, testar:

```text
T1
T2
T3
T1T2
T1T3
T2T3
```

Estado sugerido:

```text
(i, used_T1, used_T2, used_T3, used_doubles)
```

Objetivo final:

```text
used_T1 = X
used_T2 = Y
used_T3 = Z
used_doubles = 5
```

Score probabilístico inicial:

```text
T1   = pTop1
T2   = pTop2
T3   = pTop3
T1T2 = pTop1 + pTop2
T1T3 = pTop1 + pTop3
T2T3 = pTop2 + pTop3
```

O DP deve incorporar a Hard Constraint do Flamengo durante a geração/validação dos estados, nunca corrigir silenciosamente a distribuição depois.

---

# Funções XYZ prioritárias

```text
is_xyz_distribution_valid(X, Y, Z)
generate_xyz_neighbors(X, Y, Z)
generate_xyz_radius(center=(9,5,5), radius=R)
xyz_ticket(games, X, Y, Z)
xyz_distribution_backtest(...)
oracle_xyz(...)
nested_xyz_distribution_selector(...)
```

---

# Telemetria XYZ esperada

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

Comparação direta:

```text
[XYZ VS SAFE]
best_safe:
best_xyz:
delta P13+:
delta P12+:
delta mean:
```

Pairwise:

```text
wins
ties
losses
P13+ wins/ties/losses
P12+ wins/ties/losses
```

---

# OracleXYZ

Para cada concurso, usando o resultado real apenas como diagnóstico:

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
OracleDistribution P13+: 41.69%
OracleFull P13+:         41.93%
```

Se `OracleXYZ` não superar de forma relevante esses tetos, ampliar o espaço removendo Top1 provavelmente terá baixo valor marginal.

---

# Busca em raios

Não testar todo o espaço de uma vez.

```text
raio 0 → 9/5/5
raio 1 → seis vizinhos unitários
raio 2 → somente se raio 1 mostrar sinal
raio 3+ → somente com evidência nested
```

Isso reduz risco de overfitting por busca excessiva.

---

# NestedXYZDistributionSelector

Fluxo:

```text
histórico até N
      ↓
gerar espaço XYZ do raio permitido
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

---

# Top1-only e valor das cinco marcas extras

Implementações planejadas:

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

Nome:

```text
joint_probability
```

---

# Opportunity Dataset / DoubleValueModel

Dataset planejado:

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
```

Walk-forward:

```python
assert train_contest < test_contest
```

---

# Controle de experimentos

Arquivo planejado:

```text
output/experiments.csv
```

Campos:

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
- [x] benchmarks Top1;
- [x] recovery + nested recovery;
- [x] overlap/pairwise inicial;
- [x] OracleAllocator;
- [x] OracleSecondMark;
- [x] OracleFull;
- [x] regret allocator/selector/full;
- [x] DistributionBacktest seguro `14/5/0 → 14/0/5`;
- [x] OracleDistribution.

## Fase 1 — fechar diagnóstico seguro

1. [ ] `NestedDistributionSelector`;
2. [ ] `OracleDistribution usage`;
3. [ ] regret por distribuição;
4. [ ] pairwise P13+/P12+;
5. [ ] bootstrap pareado.

## Fase 2 — XYZ end-to-end

6. [ ] validar/implementar `is_xyz_distribution_valid()`;
7. [ ] validar/implementar `generate_xyz_neighbors()`;
8. [ ] validar/implementar `generate_xyz_radius()`;
9. [ ] integrar `xyz_ticket()` ao pipeline;
10. [ ] rodar `9/5/5`;
11. [ ] rodar raio 1 completo;
12. [ ] imprimir `[XYZ DISTRIBUTION BACKTEST]` em `main.py`;
13. [ ] implementar `OracleXYZ`;
14. [ ] imprimir `OracleXYZ usage`;
15. [ ] adicionar testes XYZ;
16. [ ] comparar `best_safe × best_xyz`.

## Fase 3 — validação prospectiva XYZ

17. [ ] `NestedXYZDistributionSelector` raio 1;
18. [ ] pairwise XYZ vs safe;
19. [ ] bootstrap XYZ vs safe;
20. [ ] decidir se raio 2 merece ser aberto.

## Fase 4 — valor das marcações extras

21. [ ] Top1-only;
22. [ ] Extra Mark Efficiency;
23. [ ] Oracle Capture Rate;
24. [ ] Recovery Profile.

## Fase 5 — otimização conjunta

25. [ ] `JointMarkAllocator`;
26. [ ] `joint_probability`;
27. [ ] Opportunity Dataset;
28. [ ] DoubleValueModel;
29. [ ] `joint_learned`.

## Fase 6 — robustez

30. [ ] rolling/expanding;
31. [ ] decay temporal nested;
32. [ ] reliability tables;
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

Para XYZ, acrescentar:

```text
superar o baseline seguro em nested
↓
justificar a remoção de Top1 em alguns jogos
↓
mostrar que o ganho não depende de uma única distribuição extrema
```

---

# Princípio geral

```text
Baseline seguro
      +
Oracle Decomposition
      +
DistributionBacktest seguro
      +
NestedDistributionSelector
      +
XYZ 9/5/5 ± raio controlado
      +
OracleXYZ
      +
NestedXYZDistributionSelector
      +
JointMarkAllocator / DoubleValueModel
      +
Regret / Pairwise / Bootstrap
      +
Hard Constraints
      +
Otimização de P13+
      ↓
PALPITE FINAL
```

> **Não promover a melhor regra observada; promover apenas a regra que continuar melhor quando escolhida usando somente o passado e avaliada prospectivamente.**