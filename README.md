# Loteca ML — Estratégia 9 Secos / 5 Duplos / 0 Triplos

Projeto de Machine Learning para geração de **um único palpite final da Loteca**, com foco em **maximizar a capacidade de atingir 13 ou 14 acertos**, respeitando a estrutura fixa de:

- **9 secos**;
- **5 duplos**;
- **0 triplos**;
- **19 marcações** no total.

O projeto combina probabilidades por partida, ranking Top1/Top2/Top3, backtesting histórico, walk-forward, constraints e otimização do ticket completo.

---

# Objetivo

A base histórica é lida de:

```text
data/concursos_anteriores.csv
```

O próximo concurso é lido de:

```text
data/proximo_concurso.csv
```

Para cada partida são utilizadas probabilidades normalizadas:

```text
p(1) = vitória do mandante
p(X) = empate
p(2) = vitória do visitante

p(1) + p(X) + p(2) = 1
```

O sistema deve produzir somente **um ticket final por concurso**.

---

# Ranking probabilístico

Os três resultados são ordenados da maior para a menor probabilidade:

```text
Top1 = resultado mais provável
Top2 = segundo resultado mais provável
Top3 = resultado menos provável
```

Em caso de empate probabilístico:

```text
1 > 2 > X
```

No histórico, o resultado real pode ser representado por:

```text
top1_hit
top2_hit
top3_hit
```

Isso permite medir com que frequência o resultado real ocupou cada posição do ranking.

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

Como:

```text
9 × 1 + 5 × 2 = 19
```

## Flamengo

Quando o **FLAMENGO/RJ** participar do concurso, sua vitória deve obrigatoriamente estar coberta:

```text
Flamengo mandante  → incluir 1
Flamengo visitante → incluir 2
```

A vitória pode aparecer como seco ou dentro de um duplo.

---

# Soft Constraints

## Palmeiras

Favorecer, quando o custo probabilístico for pequeno, soluções que excluam a vitória do **PALMEIRAS/SP**.

O limiar atual é:

```text
0.03
```

Esse valor deve ser tratado como parâmetro experimental e futuramente otimizado em walk-forward.

## Concentração orientada pelo histórico

A ideia de simplesmente concentrar Top1 nas primeiras posições passa a ser tratada como **hipótese histórica**, e não como vantagem presumida.

Regra conceitual:

> **Favorecer ordenações que concentrem, nas posições prioritárias, os resultados Top1, Top2 e Top3 que historicamente apresentaram maior contribuição para 13+ acertos. A preferência deve ser baseada em evidência de backtest, desempenho por posição e estabilidade fora da amostra, e não em uma ordem fixa previamente definida.**

Devem ser avaliados:

```text
Top1_hit por posição J01..J14
Top2_hit por posição J01..J14
Top3_hit por posição J01..J14
contribuição para 13+
maior run consecutiva de Top1
número de runs
número de transições Top1↔Top2↔Top3
fragmentation_score
concentração nas primeiras N posições
```

Runs longas e baixa fragmentação só devem influenciar o otimizador se demonstrarem associação consistente com 13+ fora da amostra.

---

# Estratégias atualmente implementadas

As políticas atuais são:

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

Prioriza maior:

```text
p(Top2)
```

## uncertainty

Prioriza maior:

```text
1 - p(Top1)
```

## margin

Prioriza menor separação entre Top1 e Top2:

```text
1 - (p(Top1) - p(Top2))
```

## ratio

Prioriza maior:

```text
p(Top2) / p(Top1)
```

## hist_top1

Protege com duplos as posições em que o Top1 foi historicamente menos confiável.

## hist_top2

Prioriza posições em que o Top2 apresentou maior taxa histórica de ocorrência.

As políticas históricas usam suavização Dirichlet e são recalculadas apenas com concursos anteriores em cada passo walk-forward.

## exact

Avalia exaustivamente:

```text
C(14,5) = 2.002
```

formas de escolher quais cinco partidas recebem duplo.

Para cada combinação calcula:

```text
P(14)
P(13)
P(>=13)
E[acertos]
```

Objetivo principal:

```text
argmax P(>=13)
```

### Limitação atual do exact

O `exact` atual otimiza **quais cinco jogos recebem duplo**, mas mantém estruturalmente:

```text
Seco  = Top1
Duplo = Top1 + Top2
```

Portanto, salvo constraints, a distribuição implícita continua sendo:

```text
14 marcações Top1
5 marcações Top2
0 marcações Top3
```

A evolução desejada é permitir que a própria distribuição Top1/Top2/Top3 seja otimizada.

---

# Walk-forward validation

A seleção atual de políticas é feita em **walk-forward**, evitando usar informação futura.

Fluxo:

```text
Concursos 1..N     → histórico disponível
Concurso N+1       → teste
Concursos 1..N+1   → histórico disponível
Concurso N+2       → teste
...
```

Com a base atual:

```text
445 concursos totais
30 concursos na janela histórica inicial
415 concursos de teste walk-forward
```

Em cada concurso de teste, scores históricos são calculados somente com concursos anteriores.

---

# Estado atual do backtest

Exemplo recente:

```text
[TRAIN] 445 concursos; Top hits: [0.517817, 0.265329, 0.216854]

[BACKTEST]
gain         → 14: 0 | 13: 6 | hits: 3628
uncertainty  → 14: 0 | 13: 6 | hits: 3619
margin       → 14: 0 | 13: 5 | hits: 3621
ratio        → 14: 0 | 13: 6 | hits: 3616
hist_top1    → 14: 0 | 13: 5 | hits: 3554
hist_top2    → 14: 0 | 13: 5 | hits: 3568
exact        → 14: 0 | 13: 6 | hits: 3614
```

Frequência histórica do resultado real no ranking:

```text
Top1: 51.7817%
Top2: 26.5329%
Top3: 21.6854%
```

Os resultados mostram que várias estratégias estão praticamente empatadas em 13+, tornando importante medir incerteza antes de declarar uma política superior.

---

# Próxima prioridade — backtest 10–14 completo

Antes de ampliar fortemente a complexidade do otimizador, o backtest deve passar a registrar a distribuição completa de acertos:

```text
14
13
12
11
10
<=9
```

Além de:

```text
média
mediana
desvio-padrão
mínimo
máximo
P13+ empírico
P12+ empírico
```

A seleção entre estratégias deve evoluir de um simples desempate por `hits` para uma hierarquia mais alinhada ao objetivo:

```text
14
↓
13+
↓
12+
↓
estabilidade fora da amostra
↓
média de acertos
```

Exemplo desejado:

```text
[BACKTEST]
strategy       14   13   12   11   10   <=9   P13+    P12+
gain            0    6    ?    ?    ?    ?     ...     ...
uncertainty     0    6    ?    ?    ?    ?     ...     ...
exact           0    6    ?    ?    ?    ?     ...     ...
```

Salvar também os resultados concurso a concurso em:

```text
output/backtest.csv
```

Campos sugeridos:

```text
concurso
strategy
ordering
distribution_id
hits
hit_14
hit_13
hit_12
p13_plus_empirical
p12_plus_empirical
double_games
ticket
historical_score
probability_score
similarity_score
```

---

# P13+ e P12+ empíricos

Além das probabilidades teóricas do ticket, registrar as frequências efetivamente observadas no walk-forward.

Exemplo:

```text
P13+ empírico = concursos com 13 ou 14 / concursos testados
P12+ empírico = concursos com 12, 13 ou 14 / concursos testados
```

Isso permite distinguir:

```text
P(13+) modelado
```

de:

```text
P13+ observado fora da amostra
```

As duas medidas devem permanecer separadas na telemetria.

---

# Incerteza estatística e bootstrap

Diferenças pequenas entre estratégias não devem ser interpretadas automaticamente como superioridade.

Implementar:

```text
bootstrap >= 1.000 reamostragens
IC95% de P13+
IC95% de P12+
IC95% da diferença entre estratégias
estabilidade temporal
```

Comparações prioritárias:

```text
gain vs uncertainty
gain vs ratio
gain vs exact
uncertainty vs exact
```

Exemplo desejado:

```text
[BOOTSTRAP]
gain vs exact
ΔP13+: +0.00 p.p.
IC95%: [..., ...]
Conclusão: estratégias estatisticamente indistinguíveis
```

Quando não houver evidência suficiente, o sistema deve registrar empate estatístico em vez de escolher um vencedor apenas por poucos acertos agregados.

---

# Diagnóstico de calibração

O treinamento calcula:

```text
Brier multiclasse
Log Loss
ECE
bins de calibração
position_rank_hit_rates
```

Exemplo recente:

```text
[CALIBRATION]
Brier:    0.588408
Log Loss: 0.985557
ECE:      0.012378
```

Nesta etapa essas medidas são diagnósticas e ainda não recalibram as probabilidades usadas na geração do ticket.

## Calibração aplicada

Comparar futuramente:

```text
exact_raw
exact_calibrated
```

Fluxo:

```text
probabilidades originais
        ↓
calibrador treinado somente no passado
        ↓
probabilidades calibradas
        ↓
Top1 / Top2 / Top3
        ↓
otimizador
        ↓
backtest walk-forward
```

A calibração só deverá ser adotada se melhorar desempenho fora da amostra.

---

# Métricas probabilísticas do ticket

Para cada jogo é calculada a probabilidade coberta pela marcação.

## Seco

```text
q(i) = P(resultado marcado)
```

## Duplo

```text
q(i) = P(resultado A) + P(resultado B)
```

A distribuição de acertos é tratada como Poisson-binomial, assumindo independência entre os 14 jogos.

Telemetria atual:

```text
P(14)
P(13)
P(>=13)
E[acertos]
```

A hipótese de independência deve permanecer explicitamente documentada enquanto for utilizada.

---

# Ganho marginal dos duplos

No desenho atual, transformar um seco Top1 em duplo Top1+Top2 produz:

```text
ganho marginal = p(Top2)
```

A saída registra também o ranking de ganho entre os 14 jogos.

---

# Colunas e políticas de ordenação

O projeto deve comparar diferentes ordenações e medir o impacto final sobre 13+.

## Probabilísticas

```text
ord_jogo
ord_top1_prob
ord_top2_prob
ord_top3_prob
ord_margin
ord_uncertainty
ord_entropy
ord_double_gain
```

## Históricas

```text
ord_hist_top1
ord_hist_top2
ord_hist_top3
ord_hist_position
ord_hist_13plus
```

## Adaptativas

```text
ord_knn_13plus
ord_hybrid
```

Scores de auditoria sugeridos:

```text
score_top1_prob
score_margin
score_uncertainty
score_entropy
score_double_gain
score_hist_top1
score_hist_top2
score_hist_13plus
score_knn_13plus
score_hybrid
```

---

# Historical 13+ Score

Próxima política histórica prioritária:

```text
hist_13plus
```

Em vez de perguntar apenas:

> qual posição tem mais Top1?

perguntar:

> qual combinação de posição, ranking e perfil de jogo esteve mais associada aos tickets que atingiram 13 ou 14?

Estrutura conceitual inicial:

```text
historical_13plus_score[posição][ranking]
```

Evolução possível:

```text
historical_13plus_score[
    posição,
    ranking,
    faixa_p_top1,
    faixa_margem,
    perfil_concurso
]
```

Todo score deve ser calculado somente com informação passada e validado por walk-forward.

---

# Distribution Backtest

A próxima evolução estrutural é deixar de perguntar apenas:

> quais cinco jogos recebem Top2?

para perguntar:

> como as 19 marcações devem ser distribuídas entre Top1, Top2 e Top3 para produzir mais 13+ fora da amostra?

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
19 marcações
9 secos
5 duplos
0 triplos
```

Para cada distribuição registrar:

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

O otimizador deverá poder testar:

```text
Duplo: T1T2 | T1T3 | T2T3
Seco:  T1   | T2   | T3
```

Top2/Top3 secos ou duplos que excluam Top1 só devem ser usados quando demonstrarem vantagem fora da amostra.

---

# FullMarkingOptimizer

Evolução do `exact` para otimização completa das 19 marcações:

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
Score histórico / probabilístico
   ↓
Melhor ticket
```

Comparar:

```text
historical_exact
probability_exact
hybrid_exact
```

## historical_exact

Maximiza desempenho histórico/out-of-sample para 13+.

## probability_exact

Maximiza `P(>=13)` nas probabilidades atuais.

## hybrid_exact

Combina evidência histórica e probabilidades atuais.

Forma inicial:

```text
score_total = α × score_probabilidade + (1-α) × score_histórico
```

`α` deve ser escolhido por walk-forward.

---

# Similaridade histórica / KNN

Criar:

```text
similarity_knn
```

Perfil possível do concurso:

```text
média p(Top1)
média p(Top2)
média p(Top3)
desvio p(Top1)
entropia média
margem média Top1-Top2
número de favoritos > 50%
número de jogos equilibrados
```

Fluxo:

```text
Próximo concurso
       ↓
Perfil probabilístico
       ↓
K concursos históricos mais semelhantes
       ↓
Backtest das estratégias nesse subconjunto
       ↓
Estratégia/distribuição candidata
```

A similaridade deve usar apenas dados disponíveis antes do resultado real.

---

# Ensemble de estratégias

Se `gain`, `uncertainty`, `ratio`, `exact` e futuras políticas permanecerem muito próximas, testar um ensemble em vez de forçar um vencedor único.

Exemplo conceitual:

```text
double_score(i) =
    w1 × gain_score(i)
  + w2 × uncertainty_score(i)
  + w3 × exact_preference(i)
  + w4 × hist_13plus_score(i)
```

Os pesos devem ser escolhidos por walk-forward.

O ensemble só deve ser adotado se superar os componentes individualmente fora da amostra.

---

# Estabilidade temporal

Além do desempenho agregado, medir a distribuição do sucesso ao longo do tempo.

Exemplos:

```text
primeiro terço do período
segundo terço
último terço
```

ou:

```text
janelas móveis de N concursos
```

Uma estratégia deve ser considerada mais robusta quando seu desempenho não estiver excessivamente concentrado em um único período.

Telemetria sugerida:

```text
P13+ por período
P12+ por período
média por período
desvio entre períodos
pior janela
melhor janela
```

---

# Runs e fragmentação

Métricas candidatas:

```text
longest_top1_run
number_of_top1_runs
number_of_transitions
fragmentation_score
```

Devem permanecer inicialmente como telemetria e só entrar no score final se houver evidência walk-forward.

---

# Benchmarks

Comparar novos métodos contra:

```text
gain
uncertainty
margin
ratio
exact
hist_top1
hist_top2
hist_13plus
similarity_knn
hybrid_hist_prob
exact_calibrated
5 duplos aleatórios
distribuição histórica global
```

O baseline aleatório deve usar múltiplas sementes e muitas repetições.

---

# Telemetria e auditoria

Informações atuais:

```text
p(1), p(X), p(2)
Top1/Top2/Top3
seco ou duplo
palpite
ganho marginal
ranking de ganho
constraints
P(14)
P(13)
P(>=13)
E[acertos]
Brier
Log Loss
ECE
```

Evoluções desejadas:

```text
10–14 por estratégia
P13+ empírico
P12+ empírico
intervalos de confiança
empate estatístico
distribuição T1/T2/T3
historical_13plus_score
position_score
similarity_score
probability_score
hybrid_score
fragmentation_score
estabilidade temporal
impacto quantitativo das constraints
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
│   └── predictions.csv
└── README.md
```

---

# Formato dos CSVs

```text
Delimitador: ;
Separador decimal: ,
```

A leitura aceita:

```text
UTF-8
CP1252
Latin-1
```

---

# Formato dos palpites

## Secos

```text
1
X
2
```

## Duplos

```text
1X
12
X2
```

## Triplos

```text
1X2
```

Na estratégia principal:

```text
Triplos = 0
```

---

# Execução

No PowerShell/Windows:

```powershell
python main.py
```

Com caminhos personalizados:

```powershell
python main.py --history data/concursos_anteriores.csv --next data/proximo_concurso.csv --model models/model.json --output output/predictions.csv
```

---

# Testes automatizados

Executar:

```bash
python -m unittest discover -v
```

Testes importantes:

```text
14 partidas por concurso
9 secos
5 duplos
19 marcações
0 triplos
Flamengo sempre coberto
empate do ranking 1 > 2 > X
probabilidades normalizadas
constraints preservam 9/5/0
P(13) e P(14) validados
walk-forward sem informação futura
scores históricos usando apenas informação passada
```

---

# Roadmap resumido

## Concluído

- [x] pipeline único de constraints;
- [x] invariantes 9/5/0;
- [x] políticas `gain`, `uncertainty`, `margin` e `ratio`;
- [x] `exact` com 2.002 combinações;
- [x] `P(14)`, `P(13)`, `P(>=13)` e `E[acertos]`;
- [x] ganho marginal dos duplos;
- [x] Brier multiclasse, Log Loss, ECE e bins de calibração;
- [x] matriz histórica posição × ranking;
- [x] políticas `hist_top1` e `hist_top2` com suavização;
- [x] seleção de políticas por walk-forward sem vazamento temporal.

## Próximas prioridades — ordem prática

1. [x] expandir backtest para 10–14;
2. [x] calcular `P13+` e `P12+` empíricos;
3. [x] salvar backtest concurso a concurso em `output/backtest.csv`;
4. [ ] implementar bootstrap e intervalos de confiança;
5. [ ] registrar empate estatístico entre políticas quando apropriado;
6. [ ] criar `historical_13plus_score` / `hist_13plus`;
7. [ ] implementar `distribution_backtest` Top1/Top2/Top3;
8. [ ] permitir duplos T1T3 e T2T3;
9. [ ] avaliar secos Top2/Top3;
10. [ ] implementar FullMarkingOptimizer;
11. [ ] implementar distribuição/ordenação por similaridade KNN;
12. [ ] implementar calibração aplicada e comparar `exact_raw` vs `exact_calibrated`;
13. [ ] implementar `hybrid_hist_prob` / `hybrid_exact`;
14. [ ] testar ensemble de estratégias;
15. [ ] medir estabilidade temporal e janelas móveis;
16. [ ] adicionar baseline aleatório;
17. [ ] validar runs e fragmentação;
18. [ ] substituir o desempate posicional arbitrário do `exact` por desempate neutro ou histórico validado;
19. [ ] otimizar o limiar do Palmeiras.

---

# Critério de sucesso

O projeto **não busca apenas acertar o resultado mais provável de cada partida**.

A unidade de avaliação é o ticket completo de 19 marcações.

Prioridade conceitual:

```text
14 acertos
   ↓
13 acertos
   ↓
12 acertos / estabilidade
   ↓
média de acertos
```

Toda heurística deve demonstrar ganho histórico e, preferencialmente, fora da amostra.

Quando duas estratégias forem estatisticamente indistinguíveis, o sistema deve evitar declarar superioridade sem evidência suficiente.

---

# Princípio geral

```text
Probabilidades
      +
Histórico
      +
Validação walk-forward
      +
Incerteza estatística
      +
Ordenações históricas
      +
Distribuição Top1/Top2/Top3
      +
Constraints
      +
Calibração
      +
Otimização
      ↓
PALPITE FINAL
```

A direção do projeto é tornar o sistema progressivamente menos dependente de regras intuitivas e mais orientado por evidência:

> **maximizar, de forma auditável, estatisticamente defensável e validada fora da amostra, a capacidade da aposta de atingir 13 ou 14 pontos.**
