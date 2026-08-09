# Loteca ML — Estratégia 9 Secos / 5 Duplos / 0 Triplos

Projeto de Machine Learning para geração de **um único palpite final da Loteca**, com foco em **maximizar a capacidade de atingir 13 ou 14 acertos**, respeitando:

- **9 secos**;
- **5 duplos**;
- **0 triplos**;
- **19 marcações** no total.

O projeto combina probabilidades por partida, ranking Top1/Top2/Top3, backtesting histórico, validação walk-forward, constraints e otimização do ticket completo.

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

O sistema deve produzir apenas **um ticket final por concurso**.

---

# Ranking probabilístico

```text
Top1 = resultado mais provável
Top2 = segundo resultado mais provável
Top3 = resultado menos provável
```

Desempate probabilístico:

```text
1 > 2 > X
```

No histórico, o resultado real é classificado como:

```text
top1_hit
top2_hit
top3_hit
```

Na base atual:

```text
Top1: 51.7817%
Top2: 26.5329%
Top3: 21.6854%
```

---

# Princípio central — p(Top1) é o baseline a ser batido

O `p(Top1)` é a principal referência individual de acerto do projeto.

A pesquisa histórica não deve procurar simplesmente uma métrica diferente de `p(Top1)`. O objetivo é encontrar **informação histórica incremental** capaz de identificar quando o próprio `p(Top1)` está sistematicamente superestimado ou subestimado.

Pergunta central:

> **Existe uma métrica histórica que reordene os jogos melhor que `p(Top1)` e demonstre essa vantagem fora da amostra?**

Uma nova métrica histórica só deve entrar no otimizador se:

```text
1. superar p(Top1) em walk-forward;
2. vencer nos casos em que sua ordenação discorda de p(Top1);
3. melhorar P13+ e/ou P12+ do ticket;
4. apresentar estabilidade temporal;
5. não depender de informação futura.
```

Correlação histórica isolada não é suficiente.

---

# Nova linha prioritária — confiabilidade do Top1

Em vez de substituir `p(Top1)`, o projeto deve tentar **corrigi-lo onde houver erro histórico sistemático**.

## 1. top1_residual

Definição conceitual:

```text
top1_residual = taxa_histórica_real_de_Top1 - p(Top1)_médio
```

Exemplo:

```text
Perfil A
p(Top1) médio      = 0.45
Top1 observado     = 0.52
residual           = +0.07

Perfil B
p(Top1) médio      = 0.45
Top1 observado     = 0.39
residual           = -0.06
```

Score candidato:

```text
adjusted_top1_score = p(Top1) + top1_residual
```

O objetivo é descobrir situações em que um Top1 aparentemente mais fraco é historicamente mais confiável que outro com `p(Top1)` ligeiramente maior.

---

## 2. top1_lift

Definição:

```text
top1_lift = taxa_histórica_real_de_Top1 / p(Top1)_médio
```

Interpretação:

```text
lift > 1 → Top1 historicamente subestimado
lift < 1 → Top1 historicamente superestimado
```

Score candidato:

```text
adjusted_top1_score = p(Top1) × top1_lift
```

---

## 3. conditional_top1_rate

Estimar:

```text
P(Top1_hit | contexto)
```

Contexto inicial sugerido:

```text
p(Top1)
p(Top2)
p(Top3)
margem Top1-Top2
entropia
resultado Top1 = 1/X/2
```

A posição J01..J14 deve ter prioridade menor que o contexto probabilístico, pois as políticas posicionais puras ainda não demonstraram superioridade.

---

## 4. top1_reliability

Criar uma métrica específica para responder:

> **Quanto devo confiar neste Top1?**

Primeira versão por bins:

```text
p(Top1):
33–40%
40–45%
45–50%
50–60%
60%+

margem Top1-Top2:
0–5 p.p.
5–10 p.p.
10–20 p.p.
20+ p.p.

Top1:
1 / X / 2
```

Score:

```text
top1_reliability = frequência histórica de Top1_hit no contexto
```

Os bins devem ser aprendidos somente com concursos anteriores em cada passo walk-forward.

---

# Disagreement Test

A avaliação mais importante das novas métricas históricas será feita nos casos em que elas **discordam da ordenação de `p(Top1)`**.

Considere dois jogos A e B:

```text
p(Top1_A) > p(Top1_B)
```

mas a métrica histórica produz:

```text
historical_score_A < historical_score_B
```

Esses casos devem ser registrados separadamente.

Saída desejada:

```text
[DISAGREEMENT]
metric: top1_residual
casos: N
p(top1) venceu: X
historical_score venceu: Y
empates/neutros: Z
historical_win_rate: ...
```

Uma métrica histórica só demonstra valor incremental quando consegue vencer `p(Top1)` justamente nesses casos de discordância.

---

# Meta-modelo de confiabilidade — p(top1_meta)

Além das fórmulas históricas manuais, testar um modelo secundário cujo alvo seja:

```text
top1_hit = 1 ou 0
```

Features iniciais:

```text
p_top1
p_top2
p_top3
margin_top1_top2
ratio_top2_top1
entropy
top1_is_1
top1_is_X
top1_is_2
```

Saída:

```text
p(top1_meta) = P(Top1 realmente acertar | contexto)
```

Fluxo:

```text
probabilidades originais
        ↓
contexto probabilístico
        ↓
modelo de confiabilidade
        ↓
p(top1_meta)
        ↓
ordenação / otimizador
```

O objetivo não é criar um segundo modelo que ignore `p(Top1)`, mas aprender **quando confiar mais ou menos nele**.

Comparações obrigatórias:

```text
p(top1)
vs top1_residual
vs top1_lift
vs top1_reliability
vs conditional_top1_rate
vs p(top1_meta)
```

---

# Hard Constraints

## Estrutura fixa

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

---

# Soft Constraint — Palmeiras

Favorecer, quando o custo probabilístico for pequeno, soluções que excluam a vitória do **PALMEIRAS/SP**.

Limiar atual:

```text
0.03
```

Esse valor deve ser tratado como parâmetro experimental e futuramente otimizado em walk-forward.

---

# Estratégias atualmente implementadas

```text
gain
uncertainty
margin
ratio
hist_top1
hist_top2
exact
```

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

## hist_top1

Protege com duplos as posições em que Top1 foi historicamente menos confiável.

## hist_top2

Prioriza posições em que Top2 apresentou maior frequência histórica.

As políticas históricas usam somente concursos anteriores em cada passo walk-forward.

## exact

Avalia:

```text
C(14,5) = 2.002
```

alocações de cinco duplos, maximizando principalmente:

```text
P(>=13)
```

Limitação atual:

```text
Seco  = Top1
Duplo = Top1 + Top2
```

Portanto, salvo constraints:

```text
14 marcações Top1
5 marcações Top2
0 marcações Top3
```

---

# Walk-forward validation

A seleção de políticas é feita sem informação futura:

```text
Concursos 1..N     → histórico disponível
Concurso N+1       → teste
Concursos 1..N+1   → histórico disponível
Concurso N+2       → teste
...
```

Base atual:

```text
445 concursos totais
30 concursos na janela histórica inicial
415 concursos de teste walk-forward
```

---

# Estado atual do backtest

Execução recente:

```text
gain
14: 0 | 13: 6 | 12: 17 | 11: 48 | 10: 73 | <=9: 271
P13+: 1.445783% | P12+: 5.542169% | média: 8.742169

uncertainty
14: 0 | 13: 6 | 12: 19 | 11: 48 | 10: 73 | <=9: 269
P13+: 1.445783% | P12+: 6.024096% | média: 8.720482

margin
14: 0 | 13: 5 | 12: 18 | 11: 49 | 10: 75 | <=9: 268
P13+: 1.204819% | P12+: 5.542169% | média: 8.725301

ratio
14: 0 | 13: 6 | 12: 17 | 11: 48 | 10: 75 | <=9: 269
P13+: 1.445783% | P12+: 5.542169% | média: 8.713253

hist_top1
14: 0 | 13: 5 | 12: 17 | 11: 33 | 10: 59 | <=9: 301
P13+: 1.204819% | P12+: 5.301205% | média: 8.563855

hist_top2
14: 0 | 13: 5 | 12: 20 | 11: 36 | 10: 60 | <=9: 294
P13+: 1.204819% | P12+: 6.024096% | média: 8.597590

exact
14: 0 | 13: 6 | 12: 18 | 11: 46 | 10: 76 | <=9: 269
P13+: 1.445783% | P12+: 5.783133% | média: 8.708434
```

Com o critério atual:

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

`uncertainty` é selecionada porque empata em 13+ com `gain`, `ratio` e `exact`, mas apresenta melhor P12+ entre elas.

As políticas posicionais `hist_top1` e `hist_top2` permanecem como benchmarks; até o momento não superaram as políticas probabilísticas em P13+.

---

# P13+ e P12+ empíricos

```text
P13+ empírico = concursos com 13 ou 14 / concursos testados
P12+ empírico = concursos com 12, 13 ou 14 / concursos testados
```

Essas medidas são diferentes das probabilidades teóricas Poisson-binomial do ticket e devem permanecer separadas na telemetria.

Resultados concurso a concurso são salvos em:

```text
output/backtest.csv
```

---

# Bootstrap e incerteza estatística

Diferenças pequenas entre estratégias não devem ser interpretadas automaticamente como superioridade.

Implementar:

```text
bootstrap >= 1.000 reamostragens
IC95% de P13+
IC95% de P12+
IC95% das diferenças entre estratégias
estabilidade temporal
```

Também aplicar bootstrap às novas métricas de confiabilidade do Top1 e ao Disagreement Test.

Quando não houver evidência suficiente, registrar:

```text
estatisticamente indistinguíveis
```

---

# Diagnóstico de calibração

Métricas atuais:

```text
Brier multiclasse: 0.588408
Log Loss:          0.985557
ECE:               0.012378
```

O ECE baixo estabelece um baseline probabilístico forte; por isso, qualquer correção histórica de `p(Top1)` deve demonstrar ganho real fora da amostra.

Calibração aplicada futura:

```text
exact_raw
vs
exact_calibrated
```

A calibração só deve ser mantida se melhorar o desempenho walk-forward.

---

# Métricas probabilísticas do ticket

Para cada jogo:

```text
Seco:  q(i) = P(resultado marcado)
Duplo: q(i) = P(resultado A) + P(resultado B)
```

A distribuição de acertos é modelada como Poisson-binomial, sob hipótese de independência entre as 14 partidas.

Telemetria:

```text
P(14)
P(13)
P(>=13)
E[acertos]
```

---

# Distribution Backtest

Após testar as métricas capazes de desafiar `p(Top1)`, ampliar o espaço das 19 marcações.

Testar distribuições como:

```text
T1=14 | T2=5 | T3=0
T1=13 | T2=5 | T3=1
T1=13 | T2=4 | T3=2
T1=12 | T2=6 | T3=1
...
```

Sempre respeitando:

```text
9 secos
5 duplos
0 triplos
19 marcações
```

Medir:

```text
N14
N13
N12
N11
N10
P13+ empírico
P12+ empírico
média
mediana
desvio-padrão
```

---

# Duplos flexíveis e secos alternativos

Testar futuramente:

```text
Duplo: T1T2 | T1T3 | T2T3
Seco:  T1   | T2   | T3
```

Top2/Top3 secos ou duplos que excluam Top1 só devem ser usados quando demonstrarem vantagem fora da amostra.

---

# FullMarkingOptimizer

Evolução do `exact` para otimização completa:

```text
14 jogos
   ↓
Seco:  T1 | T2 | T3
Duplo: T1T2 | T1T3 | T2T3
   ↓
Hard Constraints
   ↓
9 secos + 5 duplos
   ↓
Score probabilístico / histórico incremental
   ↓
Melhor ticket
```

Comparar:

```text
probability_exact
historical_exact
hybrid_exact
```

No modelo híbrido, o histórico só deve receber peso se demonstrar informação incremental sobre `p(Top1)`.

---

# Historical 13+ Score

`hist_13plus` permanece como linha experimental, mas deixa de ser a primeira prioridade histórica.

Pergunta:

> qual combinação de posição, ranking e perfil esteve associada a tickets com 13 ou 14?

Essa métrica deve ser comparada contra as métricas de confiabilidade do Top1 e contra `p(Top1)` puro.

Ela só deve entrar no otimizador se acrescentar sinal fora da amostra.

---

# Similaridade histórica / KNN

Criar futuramente:

```text
similarity_knn
```

Features possíveis do perfil do concurso:

```text
média p(Top1)
média p(Top2)
média p(Top3)
desvio p(Top1)
entropia média
margem média Top1-Top2
número de favoritos fortes
número de jogos equilibrados
```

A similaridade deve usar apenas informação disponível antes do resultado real.

---

# Estabilidade temporal

Medir desempenho em:

```text
primeiro terço
segundo terço
último terço
```

ou janelas móveis de N concursos.

Registrar:

```text
P13+ por período
P12+ por período
Top1 accuracy por período
Disagreement win rate por período
média por período
pior janela
melhor janela
```

---

# Telemetria desejada

Além da saída atual, registrar futuramente por jogo:

```text
p_top1
p_top2
p_top3
margin_top1_top2
entropy
top1_residual
top1_lift
top1_reliability
conditional_top1_rate
p_top1_meta
baseline_rank
historical_rank
disagreement_flag
```

E agregados:

```text
Top1 baseline accuracy
Historical metric accuracy
Disagreement cases
p(top1) wins
historical wins
historical win rate
P13+
P12+
IC95%
estabilidade temporal
```

---

# Estrutura do repositório

```text
loteca-ML-9s-5d-0t/
│
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
│   └── backtest.csv
└── README.md
```

---

# Execução

```powershell
python main.py
```

Testes:

```bash
python -m unittest discover -v
```

---

# Roadmap resumido

## Concluído

- [x] pipeline único de constraints;
- [x] invariantes 9/5/0;
- [x] políticas `gain`, `uncertainty`, `margin` e `ratio`;
- [x] `exact` com 2.002 combinações;
- [x] `P(14)`, `P(13)`, `P(>=13)` e `E[acertos]`;
- [x] Brier, Log Loss, ECE e bins de calibração;
- [x] matriz posição × ranking;
- [x] `hist_top1` e `hist_top2`;
- [x] walk-forward sem vazamento temporal;
- [x] backtest 10–14;
- [x] `P13+` e `P12+` empíricos;
- [x] `output/backtest.csv`.

## Próximas prioridades — ordem prática

1. [x] implementar **Disagreement Test** contra `p(Top1)`;
2. [x] implementar `top1_residual`;
3. [x] implementar `top1_lift`;
4. [x] implementar `top1_reliability` por bins;
5. [ ] implementar `conditional_top1_rate`;
6. [ ] implementar `p(top1_meta)`;
7. [ ] comparar todas as métricas contra `p(Top1)` em walk-forward;
8. [ ] bootstrap + IC95% das diferenças e do Disagreement Test;
9. [ ] medir estabilidade temporal das métricas candidatas;
10. [ ] registrar empate estatístico quando apropriado;
11. [ ] testar `historical_13plus_score` como métrica complementar;
12. [ ] implementar `distribution_backtest` Top1/Top2/Top3;
13. [ ] permitir duplos T1T3 e T2T3;
14. [ ] avaliar secos Top2/Top3;
15. [ ] implementar FullMarkingOptimizer;
16. [ ] implementar KNN por similaridade;
17. [ ] implementar calibração aplicada;
18. [ ] implementar `hybrid_hist_prob` / `hybrid_exact`;
19. [ ] adicionar baseline aleatório;
20. [ ] validar runs e fragmentação;
21. [ ] remover/substituir desempate posicional arbitrário do `exact`;
22. [ ] otimizar o limiar do Palmeiras.

O teste walk-forward atual mantém essas métricas apenas como telemetria: as três
ficaram abaixo de 50% de vitórias nos pares informativos de discordância e, por
isso, **não alteram o ticket final** até demonstrarem ganho incremental. Os scores
e o contexto de confiabilidade são exportados em `output/predictions.csv`.

---

# Critério de sucesso

O projeto não busca simplesmente encontrar uma heurística histórica diferente.

A nova régua é:

> **uma métrica histórica só é útil se acrescentar informação que `p(Top1)` ainda não contém.**

Critérios mínimos:

```text
superar p(Top1) fora da amostra
↓
vencer nos casos de discordância
↓
melhorar P13+ / P12+
↓
manter estabilidade temporal
↓
resistir a bootstrap / intervalos de confiança
```

A unidade final de avaliação continua sendo o ticket completo de 19 marcações.

---

# Princípio geral

```text
p(Top1) baseline
      +
Histórico incremental
      +
Teste de discordância
      +
Validação walk-forward
      +
Incerteza estatística
      +
Distribuição Top1/Top2/Top3
      +
Constraints
      +
Otimização
      ↓
PALPITE FINAL
```

> **O histórico não deve competir com `p(Top1)` por intuição; deve demonstrar, fora da amostra, onde `p(Top1)` pode ser corrigido e se essa correção aumenta a capacidade do ticket de atingir 13 ou 14 pontos.**
