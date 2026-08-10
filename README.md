# Loteca ML — Estratégia 9 Secos / 5 Duplos / 0 Triplos

Projeto de Machine Learning para gerar **um único palpite final da Loteca**, com foco em maximizar a probabilidade de atingir **13 ou 14 acertos**, respeitando sempre:

- **9 secos**;
- **5 duplos**;
- **0 triplos**;
- **19 marcações** no total.

O projeto combina probabilidades por partida, ranking Top1/Top2/Top3, walk-forward, hard/soft constraints, backtesting, oráculos diagnósticos e otimização do ticket.

> O objetivo principal não é maximizar accuracy jogo a jogo. A unidade final é o **ticket completo de 19 marcações**, com prioridade para **P(>=13)**.

---

# Objetivo e dados

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

Accuracy, média, win rate, Brier Score, Log Loss e ECE são principalmente métricas diagnósticas. Uma alteração só deve ser promovida quando melhorar o **ticket fora da amostra**.

---

# Princípio central — preservar Top1

O `p(Top1)` continua sendo o baseline individual mais forte.

Critérios mínimos para substituir/reordenar Top1:

```text
1. superar p(Top1) em walk-forward;
2. vencer quando houver discordância;
3. melhorar P13+ do ticket;
4. apresentar estabilidade temporal;
5. não usar informação futura.
```

Benchmarks atuais:

```text
[DISAGREEMENT] top1_residual
3234 casos | baseline 802 x histórico 750 | neutros 1682 | win rate 48.32%

[DISAGREEMENT] top1_lift
3264 casos | baseline 801 x histórico 755 | neutros 1708 | win rate 48.52%

[DISAGREEMENT] top1_reliability
3323 casos | baseline 821 x histórico 746 | neutros 1756 | win rate 47.61%

[TOP1-META]
Brier baseline: 0.233977
Brier meta:     0.240629

[DISAGREEMENT] p_top1_meta
4107 casos | baseline 1140 x meta 896 | neutros 2071 | win rate 44.01%
```

Conclusão:

> `top1_residual`, `top1_lift`, `top1_reliability` e `p(top1_meta)` permanecem como benchmarks/telemetria e não alteram o ticket final.

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

`exact` avalia as `C(14,5) = 2.002` posições possíveis dos cinco duplos e maximiza principalmente `P(>=13)` segundo as probabilidades disponíveis.

Distinguir:

```text
exact_probability = otimização ex-ante usando probabilidades
oracle_allocator  = diagnóstico ex-post usando resultados reais
```

## SecondMarkSelector

Candidatos:

```text
top2_baseline
recovery
threshold_recovery
second_mark_meta
double_value
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

`uncertainty` permanece selecionada por desempate operacional. Ainda não existe separação robusta em P13+ entre `gain`, `top2_probability`, `uncertainty`, `ratio` e `exact`.

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

# Error Recovery Score

```text
recovery_top2 = P(Top2_hit | Top1_miss, contexto)
recovery_top3 = P(Top3_hit | Top1_miss, contexto)
```

Resultado atual:

```text
[SECOND-MARK DISAGREEMENT]
739 casos | Top2 368 x recovery 371 | recovery win rate 50.20%
seletor final: top2_baseline
```

Thresholds:

```text
0.00 → 50.20% | IC95% 46.82%–53.86%
0.02 → 49.48% | IC95% 45.74%–53.36%
0.05 → 52.28% | IC95% 48.45%–56.47%
0.10 → 52.77% | IC95% 48.30%–57.23%
0.15 → 52.92% | IC95% 47.66%–58.19%
```

Nested recovery:

```text
Top2 baseline:   P13+ 1.4458% | P12+ 6.0241% | média 8.7205
Nested recovery: P13+ 1.2048% | P12+ 8.6747% | média 8.7759

delta P13+: -0.2410 p.p.
delta P12+: +2.6506 p.p.
```

Conclusão:

> O recovery atual melhora P12+ e média, mas reduz P13+. Portanto, `top2_baseline` permanece ativo. O problema de segunda marca continua relevante porque os oráculos mostram grande teto estrutural; o modelo atual apenas não consegue capturá-lo prospectivamente.

---

# Oracle Decomposition — implementado

Os oráculos usam resultados reais **somente para diagnóstico retrospectivo** e nunca alimentam a previsão final.

```text
[ORACLE DECOMPOSITION]

baseline
P13+:  1.45% | P12+:  6.02% | média  8.7205

allocator oracle
P13+: 11.08% | P12+: 31.08% | média 10.7229

selector oracle
P13+:  5.54% | P12+: 21.45% | média 10.1831

full oracle
P13+: 41.93% | P12+: 65.06% | média 12.0289
```

Leitura:

- `OracleAllocator`: grande teto em **onde colocar os cinco duplos**;
- `OracleSecondMark`: há valor em escolher corretamente `T1T2` versus `T1T3`;
- `OracleFull`: existe forte interação entre allocator e selector;
- a arquitetura 9/5/0 ainda está longe do teto retrospectivo.

## Regret por componente

```text
[REGRET ALLOCATOR]
média 2.0024 | zero 8.92% | 2+ 67.23% | máximo 5

[REGRET SELECTOR]
média 1.4627 | zero 18.07% | 2+ 44.34% | máximo 4

[REGRET FULL]
média 3.3084 | zero 0.96% | 2+ 95.42% | máximo 5
```

---

# DistributionBacktest — implementado

Fase segura: Top1 permanece coberto nos 14 jogos e as cinco marcações extras são distribuídas entre Top2 e Top3.

```text
14/5/0
14/4/1
14/3/2
14/2/3
14/1/4
14/0/5
```

O posicionamento dentro de cada distribuição é otimizado por probabilidade pré-jogo, sem usar o resultado real.

Resultado atual:

```text
[DISTRIBUTION BACKTEST]

14/5/0: P13+ 1.45% | P12+ 5.54% | média 8.7446
14/4/1: P13+ 0.48% | P12+ 6.02% | média 8.7446
14/3/2: P13+ 0.72% | P12+ 6.27% | média 8.7639
14/2/3: P13+ 1.20% | P12+ 5.78% | média 8.7807
14/1/4: P13+ 1.20% | P12+ 5.30% | média 8.7590
14/0/5: P13+ 1.69% | P12+ 5.30% | média 8.6940
```

Leitura atual:

```text
melhor P13+ observado: 14/0/5
melhor P12+ observado: 14/3/2
melhor média:          14/2/3
```

A vantagem histórica de `14/0/5` sobre `14/5/0` em P13+ é pequena e **não pode ser promovida sem nested/robustez estatística**.

## Constraint-aware distribution

Quando uma Hard Constraint tornar uma composição extrema inviável, registrar explicitamente:

```text
requested_distribution
effective_distribution
constraint_adjusted = true/false
```

Telemetria desejada:

```text
[DISTRIBUTION CONSTRAINT ADJUSTMENTS]
14/5/0 → ajustada em ... concursos
14/0/5 → ajustada em ... concursos
```

---

# OracleDistribution — implementado

```text
[ORACLE DISTRIBUTION]
P13+: 41.69%
P12+: 64.34%
```

Esse teto é muito próximo do `OracleFull`:

```text
OracleDistribution P13+: 41.69%
OracleFull         P13+: 41.93%
diferença:                  0.24 p.p.
```

Isso sugere que grande parte do teto da arquitetura atual pode ser expressa pela decisão conjunta de:

```text
1. quantos Top2/Top3 usar nas cinco marcas extras;
2. em quais jogos colocá-los.
```

## OracleDistribution usage

Adicionar frequência de ótimo por concurso:

```text
[ORACLE DISTRIBUTION USAGE]
14/5/0: ...
14/4/1: ...
14/3/2: ...
14/2/3: ...
14/1/4: ...
14/0/5: ...
```

Objetivo: descobrir se existe uma distribuição estruturalmente dominante ou se a distribuição ótima muda fortemente entre concursos.

## Distribution regret

Registrar também:

```text
[DISTRIBUTION REGRET]
distribution     mean_regret   median   zero   2+   max
14/5/0           ...
14/4/1           ...
14/3/2           ...
14/2/3           ...
14/1/4           ...
14/0/5           ...
```

---

# Prioridade 1 — NestedDistributionSelector

Não selecionar `14/0/5` simplesmente porque foi melhor nos mesmos 415 concursos usados na avaliação.

Fluxo obrigatório:

```text
histórico até N
      ↓
comparar as seis distribuições somente no passado
      ↓
selecionar distribuição
      ↓
congelar
      ↓
aplicar no concurso N+1
      ↓
registrar resultado
      ↓
repetir
```

Telemetria:

```text
[NESTED DISTRIBUTION]
usage 14/5/0: ...
usage 14/4/1: ...
usage 14/3/2: ...
usage 14/2/3: ...
usage 14/1/4: ...
usage 14/0/5: ...

baseline P13+: ...
nested P13+:   ...
delta P13+:    ...

baseline P12+: ...
nested P12+:   ...
delta P12+:    ...
```

Somente o nested pode promover uma distribuição diferente do baseline.

---

# Prioridade 2 — comparação pareada e bootstrap das distribuições

Comparar, no mínimo:

```text
14/0/5 vs 14/5/0
14/3/2 vs 14/5/0
14/2/3 vs 14/5/0
```

Pairwise:

```text
wins
ties
losses
mean_delta_hits
P13+ wins / ties / losses
P12+ wins / ties / losses
```

Bootstrap pareado por concurso:

```text
>= 1.000 reamostragens
IC95% de ΔP13+
IC95% de ΔP12+
IC95% de Δmean
P(A > B)
```

Diferenças de um ou dois concursos com 13+ não devem ser interpretadas como evidência forte.

---

# Top1-only baseline

Adicionar explicitamente o ticket sem nenhuma das cinco proteções:

```text
[TOP1 ONLY]
P13+:
P12+:
mean:
```

Objetivo: separar a qualidade do baseline Top1 do ganho realmente produzido pelas cinco marcações extras.

## Extra Mark Efficiency

Por concurso:

```text
extra_mark_efficiency = (hits_ticket - hits_top1_only) / 5
```

Telemetria:

```text
[EXTRA MARK EFFICIENCY]
uncertainty: ...
14/0/5: ...
joint_probability: ...
oracle_full: ...
```

---

# Oracle Capture Rate

Medir quanto do ganho possível pelas cinco marcações extras cada estratégia captura:

```text
capture_rate =
    (hits_policy - hits_top1_only)
    /
    (hits_oracle_full - hits_top1_only)
```

Quando não houver ganho oracle disponível, registrar separadamente `no_oracle_gain_available`.

Telemetria:

```text
[ORACLE CAPTURE]
policy             mean_capture   median_capture
uncertainty        ...            ...
top2_probability   ...            ...
14/0/5             ...            ...
joint_probability  ...            ...
```

---

# Recovery Profile por concurso

Adicionar:

```text
top1_hits
top1_misses
recoverable_by_top2
recoverable_by_top3
recoverable_by_either
```

Exemplo:

```text
[RECOVERY PROFILE]
Top1 hits:            8
Top1 misses:          6
recoverable by Top2:  3
recoverable by Top3:  3
recoverable by either: 6
```

Também segmentar desempenho das distribuições por força do baseline Top1:

```text
Top1 fez 10+
Top1 fez 9
Top1 fez 8
Top1 fez 7
Top1 fez <=6
```

Isso ajuda a entender em quais regimes é possível alcançar 13 com apenas cinco proteções.

---

# Opportunity Dataset

Criar:

```text
output/opportunity_dataset.csv
```

Campos por partida:

```text
contest
game
p_top1
p_top2
p_top3
gap_12
gap_23
entropy
ratio_top2_top1
ratio_top3_top1
top1_result
top2_result
top3_result
top1_hit
recoverable_by_top2
recoverable_by_top3
```

Targets locais recomendados:

```text
top1_miss
extra_gain_top2
extra_gain_top3
```

Definição:

```text
extra_gain_top2 = 1 se Top1 errou e Top2 foi o resultado real
extra_gain_top3 = 1 se Top1 errou e Top3 foi o resultado real
```

Não usar diretamente `oracle_selected_double` como target principal, porque ele depende da limitação global de cinco vagas e dos outros 13 jogos.

Separação desejada:

```text
modelo local de valor
        +
otimização global das 5 vagas
```

---

# JointMarkAllocator

Os resultados dos oráculos e do DistributionBacktest justificam testar uma arquitetura conjunta, paralela à sequência `DoubleAllocator → SecondMarkSelector`.

Para cada jogo gerar duas oportunidades:

```text
Jogo i → T1T2 → score_T2
Jogo i → T1T3 → score_T3
```

Depois selecionar exatamente **5 oportunidades**, respeitando:

```text
5 marcações extras no total
máximo 1 marca extra por jogo
Top1 preservado em todos os 14 jogos
Hard Constraints preservadas
```

A unidade de decisão passa a ser:

```text
"quanto vale T1T2 neste jogo?"
"quanto vale T1T3 neste jogo?"
```

em vez de separar rigidamente:

```text
"este jogo merece duplo?"
        ↓
"qual segunda marca usar?"
```

---

# Baseline prioritário — joint_probability

Antes de qualquer novo ML:

```text
score_T2 = p(Top2)
score_T3 = p(Top3)
```

Selecionar globalmente as cinco melhores oportunidades entre as 28 possibilidades, com no máximo uma por jogo.

Nome sugerido:

```text
joint_probability
```

Esse baseline gera dinamicamente a composição `14/x/(5-x)` sem precisar escolher primeiro uma distribuição fixa.

Comparar diretamente:

```text
uncertainty + top2_baseline
14/0/5 fixo
NestedDistributionSelector
joint_probability
```

Métricas:

```text
P13+
P12+
mean
regret
oracle_capture_rate
```

---

# DoubleValueModel

Depois do baseline `joint_probability`, aprender scores locais.

Features candidatas:

```text
p_top1
p_top2
p_top3
gap_12
gap_23
ratio_top2_top1
ratio_top3_top1
entropy
identidade Top1/Top2/Top3
posição do jogo
perfil probabilístico do concurso
```

Saídas:

```text
score_T2 ≈ P(extra_gain_top2 = 1 | contexto)
score_T3 ≈ P(extra_gain_top3 = 1 | contexto)
```

Todo treinamento, threshold e hiperparâmetro deve ser escolhido em walk-forward/nested walk-forward.

---

# Perfil do concurso para seleção dinâmica

Features agregadas candidatas:

```text
mean_p_top1
median_p_top1
mean_gap_12
mean_gap_23
n_gap12_below_5pp
n_gap23_below_3pp
mean_entropy
max_entropy
sum_p_top2
sum_p_top3
```

Somente depois de existir sinal robusto, testar `DistributionSelector` condicionado ao concurso.

Evitar usar `oracle_distribution` diretamente como target multiclass; preferir aprender `value_T2/value_T3` localmente e deixar o otimizador construir a distribuição.

---

# Matriz Distribution × Optimizer

Comparar futuramente:

```text
                      14/5/0  14/4/1  14/3/2  14/2/3  14/1/4  14/0/5

top2_probability         ...      ...      ...      ...      ...      ...
uncertainty               ...      ...      ...      ...      ...      ...
exact_probability         ...      ...      ...      ...      ...      ...
double_value_model        ...      ...      ...      ...      ...      ...
joint_probability         dinâmico — não exige distribuição fixa
joint_learned             dinâmico — não exige distribuição fixa
```

Cada célula/estratégia deve registrar:

```text
n14
n13
n12
P13+
P12+
mean
stddev
regret
oracle_capture_rate
```

---

# Estabilidade temporal e calibração

Comparar:

```text
primeiro terço
segundo terço
último terço
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

Toda escolha deve ocorrer dentro do nested walk-forward.

Diagnóstico de calibração atual:

```text
Brier multiclasse: 0.588408
Log Loss:          0.985557
ECE:               0.012378
```

Gerar reliability tables separadas de Top1/Top2/Top3 e salvar:

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

Antes disso, tentar capturar uma parcela maior do teto já identificado **sem remover Top1 dos 14 jogos**.

Arquitetura futura:

```text
Distribution/Value Model
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

No espaço seguro:

```text
Top1 coberto nos 14 jogos
exatamente 5 jogos com duas marcações
nenhum jogo com três marcações
```

Para `JointMarkAllocator`:

```text
exatamente 5 oportunidades selecionadas
máximo 1 oportunidade por jogo
cada oportunidade é Top2 ou Top3 daquele jogo
```

Em toda rotina histórica:

```python
assert train_contest < test_contest
```

Oráculos:

```text
nunca alimentar previsão final
nunca ser usados diretamente como features pré-jogo
servir para diagnóstico, teto estrutural e definição controlada de labels locais
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
requested_distribution
effective_distribution
constraint_adjusted
allocator
second_mark_selector
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

# Telemetria resumida

Ao final de cada execução:

```text
[SUMMARY]
Top1 accuracy:
Top1-only mean/P13+:
Selected distribution:
Selected allocator/optimizer:
Selected second mark:
Historical P13+:
Historical P12+:
Best experimental P13+:

Oracle allocator P13+:    11.08%
Oracle selector P13+:       5.54%
Oracle distribution P13+:  41.69%
Oracle full P13+:          41.93%

Allocator regret mean: 2.0024
Selector regret mean:  1.4627
Full regret mean:      3.3084
Oracle capture rate:   ...

Nested distribution P13+: ...
Joint probability P13+:   ...

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
- [x] P13+ e P12+ empíricos;
- [x] Disagreement Test do Top1;
- [x] benchmarks de correção do Top1;
- [x] Error Recovery Score + thresholds + IC95%;
- [x] nested recovery;
- [x] overlap entre allocators;
- [x] comparação pareada inicial;
- [x] `OracleAllocator`;
- [x] `OracleSecondMark`;
- [x] `OracleFull`;
- [x] regret allocator/selector/full;
- [x] `DistributionBacktest` seguro `14/5/0 → 14/0/5`;
- [x] otimização de posicionamento dentro de cada distribuição;
- [x] `OracleDistribution`;
- [x] quantificação do teto `OracleDistribution` vs `OracleFull`.

## Fase 1 — validar seleção de distribuição

1. [ ] implementar `NestedDistributionSelector`;
2. [ ] registrar uso nested de cada distribuição;
3. [ ] registrar `OracleDistribution usage`;
4. [ ] calcular regret por distribuição fixa;
5. [ ] registrar requested/effective distribution após constraints;
6. [ ] pairwise P13+/P12+ entre distribuições;
7. [ ] bootstrap pareado das melhores distribuições.

## Fase 2 — medir o valor real das cinco marcas extras

8. [ ] implementar `Top1-only baseline`;
9. [ ] implementar `Extra Mark Efficiency`;
10. [ ] implementar `Oracle Capture Rate`;
11. [ ] implementar `Recovery Profile` por concurso;
12. [ ] segmentar desempenho por quantidade de acertos/erros Top1.

## Fase 3 — otimização conjunta sem ML

13. [ ] implementar `JointMarkAllocator`;
14. [ ] implementar baseline `joint_probability`;
15. [ ] comparar `joint_probability` vs distribuições fixas;
16. [ ] comparar `joint_probability` vs NestedDistributionSelector;
17. [ ] medir regret e Oracle Capture do joint baseline.

## Fase 4 — aprender valor marginal

18. [ ] criar `output/opportunity_dataset.csv`;
19. [ ] adicionar `gap_12`, `gap_23`, entropia e ratios;
20. [ ] criar targets `extra_gain_top2` e `extra_gain_top3`;
21. [ ] implementar `DoubleValueModel`;
22. [ ] criar `joint_learned` usando `score_T2/score_T3` aprendidos;
23. [ ] validar tudo em walk-forward/nested.

## Fase 5 — robustez

24. [ ] comparar expanding × rolling windows;
25. [ ] testar decay temporal dentro do nested;
26. [ ] gerar reliability tables Top1/Top2/Top3;
27. [ ] salvar CSVs de calibração;
28. [ ] criar/manter `output/experiments.csv`;
29. [ ] IC95%/bootstrap final das estratégias candidatas.

## Fase 6 — expansão completa

30. [ ] permitir menos de 14 Top1 somente após evidência nested;
31. [ ] avaliar secos Top2/Top3;
32. [ ] avaliar duplo `T2T3`;
33. [ ] implementar `FullMarkingOptimizer` hierárquico;
34. [ ] comparar FullMarkingOptimizer com baseline em P13+;
35. [ ] remover/substituir desempates arbitrários do `exact`;
36. [ ] otimizar o limiar do Palmeiras usando validação adequada.

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
DistributionBacktest / OracleDistribution
      +
NestedDistributionSelector
      +
Top1-only / Oracle Capture
      +
JointMarkAllocator
      +
DoubleValueModel
      +
Nested Walk-Forward
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