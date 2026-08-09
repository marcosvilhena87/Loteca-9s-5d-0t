# Loteca ML — Estratégia 9 Secos / 5 Duplos / 0 Triplos

Projeto de Machine Learning para gerar **um único palpite final da Loteca**, com foco em maximizar a chance de atingir **13 ou 14 acertos**, respeitando sempre:

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

A hierarquia de decisão do projeto passa a ser explicitamente orientada à cauda superior:

```text
1. maior P13+
2. maior número de concursos com 13
3. maior P12+
4. maior número de concursos com 12
5. maior média de acertos
6. menor instabilidade/variância
```

Métricas como accuracy, média de acertos, win rate da segunda marca e Brier Score continuam importantes, mas funcionam principalmente como **diagnóstico**. Uma alteração só deve ser promovida quando melhorar o ticket fora da amostra.

---

# Princípio central — preservar o Top1

O `p(Top1)` continua sendo o baseline individual mais forte do projeto.

Uma métrica histórica só pode substituir ou reordenar o Top1 se demonstrar informação incremental fora da amostra.

Critérios mínimos:

```text
1. superar p(Top1) em walk-forward;
2. vencer quando sua ordenação discorda de p(Top1);
3. melhorar P13+ e/ou P12+ do ticket;
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

# Arquitetura — DoubleAllocator × SecondMarkSelector

O problema é dividido em duas decisões independentes.

## DoubleAllocator

Escolhe **quais cinco jogos recebem duplo**.

Políticas atuais:

```text
gain
uncertainty
margin
ratio
hist_top1
hist_top2
exact
```

Política adicional prioritária:

```text
top2_probability
```

Ela deve selecionar diretamente os cinco maiores `p(Top2)` e funcionar como baseline explícito para verificar o quanto as demais políticas realmente acrescentam.

## SecondMarkSelector

Escolhe **qual resultado acompanha o Top1 em cada duplo**.

Candidatos:

```text
top2_baseline
recovery
threshold_recovery
second_mark_meta
double_value
```

Fluxo:

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

---

# Estratégias atuais de alocação dos duplos

## gain

```text
score = p(Top2)
```

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

alocações possíveis dos cinco duplos e maximiza principalmente `P(>=13)` sob as probabilidades disponíveis.

No baseline atual:

```text
Seco  = Top1
Duplo = Top1 + Top2
```

## top2_probability — implementação prioritária

```text
score = p(Top2)
selecionar os 5 maiores scores
```

Mesmo sendo matematicamente equivalente ao `gain` no baseline `T1T2`, deve existir como baseline nominal para facilitar comparações, testes de equivalência e futuras alterações na definição de ganho.

---

# Estado atual do backtest

415 concursos fora da janela inicial:

```text
gain
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
14
↓
13+
↓
12+
↓
estabilidade
↓
média
```

Por esse critério, `uncertainty` permanece como política selecionada atualmente.

Importante: `gain`, `uncertainty`, `ratio` e `exact` estão empatados em P13+ no histórico atual. Portanto, a superioridade de `uncertainty` ainda deve ser tratada como **desempate operacional**, não como evidência forte de dominância.

---

# Error Recovery Score

O recovery é estimado usando somente concursos anteriores e somente jogos nos quais Top1 realmente falhou.

```text
recovery_top2 = P(Top2_hit | Top1_miss, contexto)
recovery_top3 = P(Top3_hit | Top1_miss, contexto)
```

Como Top2 e Top3 são os únicos resultados restantes quando Top1 erra:

```text
recovery_top2 + recovery_top3 ≈ 1
```

Baseline:

```text
Seco  = Top1
Duplo = Top1 + Top2
```

Comparação experimental:

```text
T1T2
vs
T1T3
```

## Second-Mark Disagreement

```text
739 casos
Top2 baseline wins: 368
recovery wins:      371
recovery win rate:  50.20%
seletor final:      top2_baseline
```

Isso representa apenas **+3 decisões líquidas** para recovery e deve ser tratado como empate prático.

---

# Threshold Recovery

```text
recovery_advantage = recovery_top3 - recovery_top2
```

Regra:

```text
T1T2 por padrão
T1T3 somente se recovery_advantage >= threshold
```

Resultados atuais:

```text
threshold   trocas   Top2   recovery   win rate   IC95%              ganho líquido
0.00          739     368      371      50.20%    46.82%–53.86%          +3
0.02          669     338      331      49.48%    45.74%–53.36%          -7
0.05          549     262      287      52.28%    48.45%–56.47%         +25
0.10          470     222      248      52.77%    48.30%–57.23%         +26
0.15          342     161      181      52.92%    47.66%–58.19%         +20
```

Todos os IC95% ainda incluem 50%. Portanto, nenhum threshold está promovido ao ticket final.

```text
SecondMarkSelector = top2_baseline
```

---

# Nested Walk-Forward para threshold

Cada threshold é escolhido usando somente o passado e congelado antes de avaliar o concurso seguinte.

```text
415 concursos de teste

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

Thresholds selecionados usando somente o passado:

```text
0.05: 373 concursos
0.10:  42 concursos
0.15:   0 concursos
```

O nested melhora P12+ e média, mas reduz P13+. Portanto, **não foi promovido**.

Esse resultado reforça um princípio do projeto:

> uma regra pode melhorar decisões individuais ou a média de acertos e ainda piorar a cauda de 13+.

---

# Nova prioridade — Oracle Decomposition

Antes de aumentar a complexidade do ML, medir o **teto de melhoria disponível** em cada parte da arquitetura.

Como existem apenas:

```text
C(14,5) = 2.002
```

combinações de cinco duplos por concurso, é viável testar exaustivamente todas as escolhas no backtest.

Para 415 concursos:

```text
415 × 2.002 = 830.830 combinações
```

## Oracle A — OracleAllocator

Mantém:

```text
Seco  = Top1
Duplo = Top1 + Top2
```

mas usa o resultado real apenas para fins de diagnóstico retrospectivo e encontra quais cinco jogos deveriam ter recebido duplo.

Objetivo:

> medir o teto máximo disponível para melhorar o `DoubleAllocator`.

## Oracle B — OracleSecondMark

Mantém os cinco duplos escolhidos pela política real, porém seleciona retrospectivamente `Top2` ou `Top3` em cada duplo.

Objetivo:

> medir o teto máximo disponível para melhorar o `SecondMarkSelector`.

## Oracle C — OracleFull

Pode escolher retrospectivamente:

```text
quais 5 jogos recebem duplo
+
Top2 ou Top3 em cada duplo
```

Objetivo:

> medir o teto conjunto da arquitetura atual `Top1 + 5 duplos`.

## Telemetria Oracle

```text
[ORACLE DECOMPOSITION]

baseline_mean:
allocator_oracle_mean:
selector_oracle_mean:
full_oracle_mean:

baseline_P13+:
allocator_oracle_P13+:
selector_oracle_P13+:
full_oracle_P13+:

baseline_P12+:
allocator_oracle_P12+:
selector_oracle_P12+:
full_oracle_P12+:
```

Os oráculos **jamais** participam da previsão final. Eles são apenas instrumentos diagnósticos para descobrir onde existe capacidade de melhoria.

---

# Regret do DoubleAllocator

Para cada concurso:

```text
regret = hits_oracle_allocator - hits_policy
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

Exemplo:

```text
[ALLOCATOR REGRET]
policy        mean   median   regret=0   regret=1   regret>=2
gain           ...     ...       ...        ...         ...
uncertainty    ...     ...       ...        ...         ...
margin         ...     ...       ...        ...         ...
ratio          ...     ...       ...        ...         ...
exact          ...     ...       ...        ...         ...
```

Essa análise responde diretamente:

> **quanto o ticket perde por selecionar os cinco jogos errados para receber duplo?**

---

# Similaridade entre allocators

Muitas políticas podem produzir praticamente os mesmos cinco duplos. Medir a sobreposição média entre pares:

```text
overlap(A,B) = quantidade de jogos em comum entre os 5 duplos
```

Telemetria:

```text
[ALLOCATOR OVERLAP]
uncertainty x gain             ... / 5
uncertainty x ratio            ... / 5
uncertainty x exact            ... / 5
uncertainty x top2_probability ... / 5
```

Se duas políticas apresentarem overlap próximo de 5 e resultados equivalentes, tratá-las como redundantes.

---

# Comparação pareada entre estratégias

Resultados agregados podem esconder diferenças concurso a concurso.

Para cada par de políticas, registrar:

```text
A > B
A = B
A < B
mean_delta_hits
```

Exemplo:

```text
[PAIRWISE] uncertainty vs gain
wins:   ...
ties:   ...
losses: ...
mean delta: ...
```

Para o evento principal, transformar cada concurso em:

```text
1 = ticket fez >=13
0 = ticket não fez >=13
```

Essa representação permite comparação pareada específica da cauda superior.

---

# Backtest matricial — Allocator × SecondMarkSelector

Comparar sistematicamente:

```text
                    Top2   Rec .05   Rec .10   Rec .15   Nested

gain                 ...      ...       ...       ...      ...
uncertainty          ...      ...       ...       ...      ...
margin               ...      ...       ...       ...      ...
ratio                ...      ...       ...       ...      ...
exact                ...      ...       ...       ...      ...
top2_probability     ...      ...       ...       ...      ...
```

Cada célula deve registrar:

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

Isso separa claramente:

```text
qual allocator escolhe melhor os cinco duplos?
qual selector escolhe melhor T2/T3?
qual combinação maximiza 13+?
```

---

# Bootstrap e significância

Usar comparação **pareada por concurso**.

Implementar pelo menos:

```text
bootstrap >= 1.000 reamostragens
IC95% de ΔP13+
IC95% de ΔP12+
IC95% de Δmean
probabilidade empírica de A > B
```

Para segunda marcação:

```text
IC95% de net_recovery_gain
IC95% do Second-Mark win rate
```

Quando o intervalo incluir o baseline, registrar:

```text
estatisticamente indistinguível
```

Devido ao baixo número atual de concursos com 13+, diferenças de um ou dois concursos não devem ser interpretadas como evidência forte.

---

# Segmentações prioritárias

## gap_12

```text
gap_12 = p(Top1) - p(Top2)
```

Faixas iniciais:

```text
0–3 p.p.
3–5 p.p.
5–10 p.p.
10–20 p.p.
20+ p.p.
```

Objetivo: verificar em quais regimes o duplo tem maior valor marginal.

## gap_23

```text
gap_23 = p(Top2) - p(Top3)
```

Faixas:

```text
0–2 p.p.
2–5 p.p.
5–10 p.p.
10+ p.p.
```

Hipótese:

> recovery pode agregar mais quando Top2 e Top3 são probabilisticamente próximos.

## Recovery × gap_23

```text
recovery_advantage >= R
AND
gap_23 <= G
```

Grid inicial:

```text
R ∈ {0.05, 0.10, 0.15}
G ∈ {0.02, 0.05, 0.10}
```

A seleção deve ocorrer dentro do nested walk-forward.

## p(Top1)

```text
33–40%
40–45%
45–50%
50–60%
60%+
```

## Entropia

```text
H = -Σ p(i) × log(p(i))
```

A entropia permite distinguir um jogo genuinamente equilibrado em três resultados de um jogo com apenas Top1 e Top2 muito próximos.

---

# Double Value Score

Uma formulação futura pode estimar explicitamente o valor esperado da marcação extra:

```text
double_value ≈ P(Top1_miss) × P(second_mark_hit | Top1_miss, contexto)
```

Ou comparar scores combinados:

```text
score_T2 = α × p(Top2) + (1-α) × recovery_top2
score_T3 = α × p(Top3) + (1-α) × recovery_top3
```

Grid inicial:

```text
α ∈ {0.00, 0.25, 0.50, 0.75, 1.00}
```

Qualquer `α`, threshold, janela ou feature deve ser escolhido somente usando informação disponível no passado de cada passo nested.

---

# Estabilidade temporal

Comparar desempenho em:

```text
primeiro terço
segundo terço
último terço
```

E futuramente:

```text
expanding window
rolling 50
rolling 100
rolling 200
```

Também pode ser testado decay temporal:

```text
half-life 25 concursos
half-life 50
half-life 100
half-life 200
sem decay
```

A seleção da janela ou decay deve ocorrer em nested walk-forward. Não selecionar retrospectivamente a configuração vencedora no mesmo período de teste.

---

# Calibração

Diagnóstico atual:

```text
Brier multiclasse: 0.588408
Log Loss:          0.985557
ECE:               0.012378
```

A calibração permanece diagnóstica até demonstrar ganho no ticket em walk-forward.

## Reliability por faixa

Adicionar relatório separado para Top1, Top2 e Top3:

```text
probabilidade prevista | frequência observada
30–35%                  | ...
35–40%                  | ...
40–45%                  | ...
45–50%                  | ...
50–60%                  | ...
60–70%                  | ...
70%+                     | ...
```

Salvar também:

```text
output/calibration_top1.csv
output/calibration_top2.csv
output/calibration_top3.csv
```

---

# Hard Constraints

Todo ticket deve conter exatamente:

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

As constraints devem valer em treino, backtest e previsão final.

## Palmeiras — Soft Constraint

Favorecer, quando o custo probabilístico for pequeno, soluções que excluam a vitória do **PALMEIRAS/SP**.

Limiar atual:

```text
0.03
```

O limiar é experimental e só deve ser alterado mediante validação walk-forward.

---

# Testes automatizados obrigatórios

Garantir permanentemente:

```text
14 jogos por concurso
9 secos
5 duplos
0 triplos
19 marcações
Top1 coberto no baseline
vitória do Flamengo coberta
probabilidades somando 1
Top1/Top2/Top3 distintos
desempate 1 > 2 > X
nenhum vazamento temporal
```

Em toda rotina histórica:

```python
assert train_contest < test_contest
```

Também testar equivalência entre `gain` e `top2_probability` enquanto `gain = p(Top2)`.

---

# Controle de experimentos

Criar:

```text
output/experiments.csv
```

Campos sugeridos:

```text
timestamp
model
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

Isso permite reproduzir resultados e evita perder o contexto de qual versão produziu cada métrica.

---

# Telemetria resumida

Ao final de cada execução, imprimir um bloco compacto:

```text
[SUMMARY]

Top1 accuracy:
Selected allocator:
Selected second mark:
Historical P13+:
Historical P12+:
Best experimental P13+:

Oracle allocator P13+:
Oracle selector P13+:
Oracle full P13+:

Current contest P14:
Current contest P13:
Current contest P13+:
Current contest E[hits]:
```

Manter os detalhes completos acima desse resumo para debugging.

---

# Distribution Backtest e FullMarkingOptimizer

Essas etapas ficam **depois** da validação da arquitetura atual.

Espaço futuro:

```text
Seco:  T1 | T2 | T3
Duplo: T1T2 | T1T3 | T2T3
```

Antes de abrir completamente esse espaço, o projeto deve quantificar com o `OracleFull` quanto potencial existe na arquitetura atual e demonstrar sinal robusto fora da amostra.

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
- [x] invariantes 9/5/0;
- [x] 19 marcações;
- [x] políticas `gain`, `uncertainty`, `margin`, `ratio`, `hist_top1`, `hist_top2` e `exact`;
- [x] `P(14)`, `P(13)`, `P(>=13)` e `E[acertos]`;
- [x] walk-forward sem vazamento temporal;
- [x] backtest 10–14;
- [x] P13+ e P12+ empíricos;
- [x] `output/backtest.csv`;
- [x] Disagreement Test do Top1;
- [x] `top1_residual`, `top1_lift`, `top1_reliability`;
- [x] `p(top1_meta)`;
- [x] evidência para congelar correções do Top1;
- [x] `error_recovery_score`;
- [x] Second-Mark Disagreement;
- [x] thresholds `0.00`, `0.02`, `0.05`, `0.10`, `0.15`;
- [x] IC95% por threshold;
- [x] nested walk-forward para threshold;
- [x] `net_recovery_gain`;
- [x] segmentação inicial por `gap_23`;
- [x] evidência para manter `top2_baseline` como SecondMarkSelector final.

## Fase 1 — diagnóstico estrutural

1. [ ] implementar `top2_probability` como baseline explícito;
2. [ ] medir overlap entre allocators;
3. [ ] implementar comparação pareada concurso a concurso;
4. [ ] implementar `OracleAllocator`;
5. [ ] implementar `OracleSecondMark`;
6. [ ] implementar `OracleFull`;
7. [ ] implementar regret por allocator;
8. [ ] reportar teto de P13+/P12+ de cada oracle.

## Fase 2 — avaliação conjunta do ticket

9. [ ] implementar matriz `Allocator × SecondMarkSelector`;
10. [ ] backtest ticket-level de `Top2`, `Rec .05`, `Rec .10`, `Rec .15` e `Nested`;
11. [ ] bootstrap pareado por concurso;
12. [ ] IC95% de `ΔP13+`, `ΔP12+` e `Δmean`;
13. [ ] teste pareado do evento `>=13`;
14. [ ] medir estabilidade temporal das políticas.

## Fase 3 — features de valor marginal

15. [ ] adicionar `gap_12`;
16. [ ] completar regra bidimensional `recovery_advantage × gap_23`;
17. [ ] segmentar por faixa de `p(Top1)`;
18. [ ] adicionar entropia;
19. [ ] implementar `double_value_score`;
20. [ ] melhorar `recovery_context` incrementalmente;
21. [ ] implementar `second_mark_meta` somente se houver sinal suficiente.

## Fase 4 — robustez temporal e calibração

22. [ ] comparar expanding × rolling windows;
23. [ ] testar decay temporal dentro de nested walk-forward;
24. [ ] gerar reliability tables de Top1/Top2/Top3;
25. [ ] salvar CSVs de calibração;
26. [ ] criar `output/experiments.csv`;
27. [ ] incluir hash/commit do Git nos experimentos.

## Fase 5 — expansão do espaço de marcações

28. [ ] implementar `distribution_backtest`;
29. [ ] avaliar `T2T3` somente se Oracle/validation justificar;
30. [ ] avaliar secos Top2/Top3 somente se houver evidência fora da amostra;
31. [ ] implementar `FullMarkingOptimizer`;
32. [ ] remover/substituir desempates arbitrários do `exact`;
33. [ ] otimizar o limiar do Palmeiras usando validação adequada.

---

# Critério de promoção de uma estratégia

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

Melhorar apenas P12+, média, accuracy ou win rate individual **não é suficiente** para promoção quando P13+ piora.

---

# Princípio geral

```text
p(Top1) preservado
      +
DoubleAllocator validado
      +
SecondMarkSelector validado
      +
Oracle Decomposition
      +
Regret / comparação pareada
      +
Nested Walk-Forward
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
