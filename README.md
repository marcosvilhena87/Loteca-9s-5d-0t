# Loteca ML — Estratégia 9 Secos / 5 Duplos / 0 Triplos

Projeto de Machine Learning voltado à geração de **um único palpite final da Loteca**, com foco em **maximizar a capacidade de atingir 13 ou 14 acertos**, respeitando a estrutura fixa de:

- **9 secos**;
- **5 duplos**;
- **0 triplos**;
- **19 marcações** no total.

O projeto combina probabilidades por partida, ranking Top1/Top2/Top3, backtesting histórico, constraints e otimização do ticket completo.

---

# Objetivo

A base histórica é lida a partir de:

```text
data/concursos_anteriores.csv
```

O próximo concurso é lido de:

```text
data/proximo_concurso.csv
```

Para cada partida são utilizadas as probabilidades:

```text
p(1) = vitória do mandante
p(X) = empate
p(2) = vitória do visitante
```

Essas probabilidades são normalizadas para que:

```text
p(1) + p(X) + p(2) = 1
```

---

# Ranking probabilístico

Os três resultados são ordenados da maior para a menor probabilidade:

```text
Top1 = resultado mais provável
Top2 = segundo resultado mais provável
Top3 = resultado menos provável
```

Em caso de empate probabilístico, é utilizado o critério:

```text
1 > 2 > X
```

O resultado real histórico pode ser representado por:

```text
top1_hit
top2_hit
top3_hit
```

permitindo medir com que frequência o resultado real ocupou cada posição do ranking.

---

# Hard Constraints

## Estrutura fixa

Todo ticket deve conter exatamente:

```text
9 secos
5 duplos
0 triplos
```

Total:

```text
9 × 1 + 5 × 2 = 19 marcações
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

Na implementação atual, a substituição de um seco vencedor do Palmeiras ocorre somente quando a perda de probabilidade para a melhor alternativa é de, no máximo:

```text
0.03
```

Esse valor deve ser tratado como parâmetro experimental e futuramente validado por backtest walk-forward.

## Concentração orientada pelo histórico

A ideia original de simplesmente concentrar Top1 nas primeiras posições passa a ser tratada como **hipótese histórica**, e não como vantagem presumida.

A regra desejada é:

> **Favorecer ordenações que concentrem, nas posições prioritárias, os resultados Top1, Top2 e Top3 que historicamente apresentaram maior contribuição para 13+ acertos. A preferência deve ser baseada em evidência de backtest, desempenho por posição e estabilidade fora da amostra, e não em uma ordem fixa previamente definida.**

Devem ser avaliados historicamente:

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

As políticas atuais para escolher os cinco jogos que recebem duplo são:

```text
gain
uncertainty
margin
ratio
exact
hist_top1
hist_top2
```

As políticas históricas são avaliadas exclusivamente em **walk-forward**. `hist_top1`
protege com duplos as cinco posições em que o Top1 foi menos confiável no passado;
`hist_top2` cobre as cinco posições com maior taxa histórica de Top2. As frequências
usam suavização Dirichlet e, em cada concurso de teste, são recalculadas somente com
os concursos anteriores.

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

O objetivo principal é:

```text
argmax P(>=13)
```

com desempates por `P(14)`, acertos esperados e critério determinístico.

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

A próxima evolução importante é permitir que a própria distribuição Top1/Top2/Top3 seja otimizada.

---

# Estado atual do backtest

Com a base atual há **445 concursos**, dos quais 415 compõem a avaliação
walk-forward após a janela histórica inicial de 30 concursos.

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

Esses números descrevem a base atual e podem mudar com a inclusão de novos concursos.

---

# Diagnóstico de calibração

O treinamento atual calcula métricas de qualidade probabilística:

```text
Brier multiclasse
Log Loss
ECE
bins de calibração
matriz posição × ranking
```

Exemplo recente:

```text
[CALIBRATION]
Brier:    0.588408
Log Loss: 0.985557
ECE:      0.012378
```

O `model.json` registra:

```text
multiclass_brier
log_loss
ece
calibration_bins
position_rank_hit_rates
```

Nesta etapa essas medidas são **diagnósticas**. Elas ainda não recalibram as probabilidades usadas na geração do ticket.

## Próxima evolução da calibração

Comparar explicitamente:

```text
exact_raw
exact_calibrated
```

Fluxo desejado:

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
P(13+) / backtest walk-forward
```

Nenhuma calibração deve usar informação futura.

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

A telemetria atual exibe:

```text
P(14)
P(13)
P(>=13)
E[acertos]
```

Exemplo:

```text
[METRIC] P(14): 0.036926%
[METRIC] P(13): 0.413255%
[METRIC] P(>=13): 0.450181%
[METRIC] E[acertos]: 8.1436
```

A hipótese de independência deve permanecer explicitamente documentada enquanto for utilizada.

---

# Ganho marginal dos duplos

O ganho marginal de transformar um seco Top1 em duplo Top1+Top2 é:

```text
ganho = p(Top2)
```

Exemplo:

```text
Top1 = 0.358197
Top2 = 0.351464

Cobertura seco  = 35.8197%
Cobertura duplo = 70.9661%
Ganho marginal  = +35.1464 p.p.
```

A saída registra também o ranking do jogo entre os 14 candidatos a duplo.

---

# Colunas e políticas de ordenação

O projeto deve comparar diferentes colunas de ordenação, sempre medindo o impacto final sobre 13+.

## Ordenações probabilísticas

```text
ord_jogo          → posição original J01..J14
ord_top1_prob     → maior p(Top1)
ord_top2_prob     → maior p(Top2)
ord_top3_prob     → maior p(Top3)
ord_margin        → maior ou menor margem Top1-Top2, conforme estratégia
ord_uncertainty   → maior incerteza
ord_entropy       → maior entropia
ord_double_gain   → maior ganho marginal do duplo
```

## Ordenações históricas

```text
ord_hist_top1     → maior Top1_hit histórico por posição/perfil
ord_hist_top2     → maior Top2_hit histórico
ord_hist_top3     → maior Top3_hit histórico
ord_hist_position → score histórico conjunto posição × ranking
ord_hist_13plus   → maior contribuição histórica para tickets de 13/14
```

## Ordenações adaptativas

```text
ord_knn_13plus    → desempenho nos concursos históricos mais semelhantes
ord_hybrid        → combinação de probabilidade atual + histórico
```

Além da posição ordenada, o CSV/debug deverá preservar os scores que originaram a ordenação:

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

Uma das próximas linhas prioritárias é criar um score diretamente alinhado ao objetivo final.

Em vez de perguntar somente:

> qual posição tem mais Top1?

perguntar:

> qual combinação de posição, ranking e perfil de jogo esteve mais associada aos tickets que atingiram 13 ou 14?

Estrutura conceitual:

```text
historical_13plus_score[posição][ranking]
```

ou, de forma mais rica:

```text
historical_13plus_score[
    posição,
    ranking,
    faixa_p_top1,
    faixa_margem,
    perfil_concurso
]
```

Esse score deve ser calculado somente com informação passada e validado por walk-forward.

---

# Nova linha prioritária — distribuição das 19 marcações

A principal evolução planejada é deixar de perguntar apenas:

> quais cinco jogos recebem Top2?

para perguntar:

> como as 19 marcações devem ser distribuídas entre Top1, Top2 e Top3 para produzir mais 13+ historicamente?

## Distribution Backtest

Criar um módulo `distribution_backtest` que avalie distribuições viáveis, por exemplo:

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
N13+
N12
média de acertos
mediana
variância / desvio-padrão
```

A distribuição historicamente vencedora deve ser validada fora da amostra.

---

# Duplos flexíveis e secos alternativos

A evolução do otimizador deve permitir:

```text
Duplo: T1T2 | T1T3 | T2T3
Seco:  T1   | T2   | T3
```

Top2 ou Top3 secos e duplos que excluam Top1 não devem ser usados por preferência subjetiva. Devem entrar apenas quando demonstrarem ganho em validação histórica fora da amostra.

---

# FullMarkingOptimizer

A evolução natural do `exact` é um otimizador completo das 19 marcações.

Fluxo desejado:

```text
14 jogos
   ↓
Opções por jogo
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

Três versões devem ser comparadas:

```text
historical_exact
probability_exact
hybrid_exact
```

## historical_exact

Maximiza empiricamente o desempenho histórico para 13+.

## probability_exact

Maximiza `P(>=13)` usando as probabilidades do concurso atual.

## hybrid_exact

Combina evidência histórica e probabilidades atuais.

Uma forma inicial de combinação poderá ser:

```text
score_total = α × score_probabilidade + (1-α) × score_histórico
```

O valor de `α` deve ser escolhido por walk-forward, e não manualmente.

---

# Distribuição histórica por posição

A matriz `position_rank_hit_rates`, já calculada como diagnóstico, deve evoluir para uma feature testável.

Estrutura:

```text
          Top1    Top2    Top3
J01        ...     ...     ...
J02        ...     ...     ...
...
J14        ...     ...     ...
```

Próximas políticas candidatas:

```text
hist_top1
hist_top2
hist_top3
hist_position
hist_13plus
```

A matriz não deve alimentar o ticket final antes de validação walk-forward.

---

# Distribuição histórica por similaridade

Uma alternativa adaptativa é escolher a distribuição usando apenas concursos históricos probabilisticamente semelhantes ao próximo.

## Perfil do concurso

Características possíveis:

```text
média p(Top1)
média p(Top2)
média p(Top3)
desvio de p(Top1)
entropia média
margem média Top1-Top2
número de favoritos > 50%
número de jogos muito equilibrados
```

## KNN histórico

Fluxo:

```text
Próximo concurso
       ↓
Perfil probabilístico
       ↓
K concursos históricos mais semelhantes
       ↓
Backtest das distribuições/ordenações nesses K concursos
       ↓
Estratégia com mais 13+
```

A similaridade deve utilizar exclusivamente informações disponíveis antes do resultado real.

---

# Runs e fragmentação

Métricas candidatas:

```text
longest_top1_run
number_of_top1_runs
number_of_transitions
fragmentation_score
```

Essas métricas devem ser tratadas inicialmente como **telemetria**.

Só devem ser promovidas a critério de desempate ou otimização se houver evidência robusta de associação com 13+ em walk-forward.

---

# Walk-forward validation

Toda nova estratégia histórica deve ser validada simulando o uso real no tempo.

Fluxo:

```text
Concursos 1..N     → treinamento
Concurso N+1       → teste
Concursos 1..N+1   → treinamento
Concurso N+2       → teste
...
```

Objetivos:

- evitar vazamento temporal;
- reduzir overfitting;
- medir desempenho verdadeiramente prospectivo;
- escolher políticas, distribuições, pesos e hiperparâmetros apenas com informação passada.

## Implementação atual

O treinamento reserva os 30 primeiros concursos como janela histórica inicial e
avalia todos os concursos seguintes em ordem cronológica. Em cada passo, as políticas
probabilísticas, `exact`, `hist_top1` e `hist_top2` são comparadas no mesmo concurso;
os scores históricos são reconstruídos apenas com o prefixo já observado. O
`model.json` registra o tamanho da janela, o número de testes fora da amostra e a
matriz histórica final usada para gerar o próximo ticket.

---

# Backtest completo

Expandir o backtest para registrar:

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

Salvar resultados concurso a concurso em:

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
double_games
ticket
historical_score
probability_score
similarity_score
```

---

# Benchmarks

Comparar qualquer novo método contra:

```text
gain
uncertainty
margin
ratio
exact
hist_top1
hist_top2
hist_position
hist_13plus
similarity_knn
hybrid_hist_prob
exact_calibrated
5 duplos aleatórios
distribuição histórica global
```

O baseline aleatório deve utilizar múltiplas sementes e muitas repetições para estimar a distribuição de desempenho por acaso.

---

# Incerteza estatística

Diferenças pequenas no número de concursos com 13+ não devem ser interpretadas automaticamente como superioridade.

Planejado:

```text
bootstrap >= 1.000 reamostragens
intervalos de confiança
comparação de diferenças de P13+
estabilidade temporal
```

---

# Telemetria e auditoria

A saída deve permitir reconstruir a decisão do algoritmo.

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
distribuição T1/T2/T3
historical_score
historical_13plus_score
position_score
similarity_score
probability_score
hybrid_score
fragmentation_score
longest_top1_run
impacto quantitativo das constraints
```

Exemplo desejado por jogo:

```text
Hist Top1 position: 0.xxx
Hist Top2 position: 0.xxx
Hist 13+ score:     0.xxx
Similarity score:   0.xxx
Probability score:  0.xxx
Final score:        0.xxx
```

---

# Estrutura do repositório

```text
loteca-ML-9s-5d-0t/
│
├── main.py
│
├── data/
│   ├── concursos_anteriores.csv
│   └── proximo_concurso.csv
│
├── scripts/
│   ├── common.py
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── predict_results.py
│
├── models/
│   └── model.json
│
├── output/
│   └── predictions.csv
│
└── README.md
```

---

# Formato dos CSVs

Os arquivos usam:

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
calibração sem vazamento temporal
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
- [x] matriz histórica posição × ranking para auditoria.
- [x] políticas `hist_top1` e `hist_top2` com suavização;
- [x] seleção de políticas por backtest walk-forward sem vazamento temporal;

## Próximas prioridades

1. [ ] criar `historical_13plus_score` / `hist_13plus`;
2. [ ] substituir o desempate posicional arbitrário do `exact` por score histórico validado ou desempate neutro;
3. [ ] implementar `distribution_backtest` Top1/Top2/Top3;
4. [ ] permitir duplos T1T3 e T2T3;
5. [ ] avaliar secos Top2/Top3;
6. [ ] implementar distribuição/ordenação por similaridade KNN;
7. [ ] implementar `hybrid_hist_prob` / `hybrid_exact`;
8. [ ] implementar calibração aplicada e comparar `exact_raw` vs `exact_calibrated`;
9. [ ] implementar FullMarkingOptimizer;
10. [ ] expandir backtest para 10–14 e `output/backtest.csv`;
11. [ ] adicionar baseline aleatório;
12. [ ] bootstrap e intervalos de confiança;
13. [ ] validar runs e fragmentação;
14. [ ] otimizar o limiar do Palmeiras.

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

Toda heurística deve demonstrar ganho histórico e, idealmente, fora da amostra.

---

# Princípio geral

```text
Probabilidades
      +
Histórico
      +
Ordenações históricas
      +
Distribuição Top1/Top2/Top3
      +
Constraints
      +
Calibração
      +
Backtest walk-forward
      +
Otimização
      ↓
PALPITE FINAL
```

A direção do projeto é tornar o sistema progressivamente menos dependente de regras intuitivas e mais orientado por evidência:

> **maximizar, de forma auditável e validada fora da amostra, a capacidade da aposta de atingir 13 ou 14 pontos.**
