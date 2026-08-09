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
p(1)  = vitória do mandante
p(X)  = empate
p(2)  = vitória do visitante
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

Exemplo:

```text
p(1) = 0.40
p(X) = 0.20
p(2) = 0.40

Top1 = 1
Top2 = 2
Top3 = X
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

## Concentração de Top1

A ideia original de concentrar Top1 nas primeiras posições passa a ser tratada como **hipótese histórica**, e não como vantagem presumida.

Devem ser avaliados historicamente:

```text
Top1_hit por posição J01..J14
maior run consecutiva de Top1
número de runs
número de transições Top1↔Top2↔Top3
fragmentation_score
concentração de Top1 nas primeiras N posições
```

A preferência por runs longas e baixa fragmentação só deve permanecer se demonstrar associação consistente com 13+ fora da amostra.

---

# Estratégias atualmente implementadas

As políticas atuais para escolher os cinco jogos que recebem duplo são:

```text
gain
uncertainty
margin
ratio
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

Com a base atual são avaliados **445 concursos**.

Exemplo recente:

```text
[TRAIN] 445 concursos; Top hits: [0.517817, 0.265329, 0.216854]

[BACKTEST]
gain         → 14: 0 | 13: 7 | hits: 3898
uncertainty  → 14: 0 | 13: 8 | hits: 3887
margin       → 14: 0 | 13: 7 | hits: 3891
ratio        → 14: 0 | 13: 8 | hits: 3885
exact        → 14: 0 | 13: 8 | hits: 3882
```

Frequência histórica do resultado real no ranking:

```text
Top1: 51.7817%
Top2: 26.5329%
Top3: 21.6854%
```

Esses números descrevem a base atual e podem mudar com a inclusão de novos concursos.

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

# Duplos flexíveis

A evolução do otimizador deve permitir:

```text
Top1 + Top2
Top1 + Top3
Top2 + Top3
```

em vez de obrigar todo duplo a ser Top1+Top2.

Isso permitirá testar empiricamente se sacrificar uma marcação Top1 ou Top2 em favor de Top3 aumenta a ocorrência de 13+.

---

# Secos alternativos

Também poderá ser permitido, de forma controlada e historicamente validada:

```text
Seco Top1
Seco Top2
Seco Top3
```

Top2 ou Top3 secos não devem ser utilizados por heurística subjetiva; devem entrar apenas quando uma estratégia treinada fora da amostra demonstrar vantagem.

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

---

# Distribuição histórica global

Baseline simples:

1. testar todas as distribuições viáveis no histórico;
2. escolher a que mais produziu 13/14;
3. aplicar essa distribuição aos concursos futuros.

É útil como benchmark, mas pode sofrer overfitting e não se adaptar ao perfil probabilístico do próximo concurso.

---

# Distribuição histórica por posição

Construir uma matriz:

```text
          Top1    Top2    Top3
J01        ...     ...     ...
J02        ...     ...     ...
...
J14        ...     ...     ...
```

Para cada posição medir:

```text
Top1_hit
Top2_hit
Top3_hit
```

Isso permite testar se determinadas posições históricas são mais adequadas para:

```text
seco Top1
seco Top2
duplo T1T2
duplo T1T3
duplo T2T3
```

Qualquer viés posicional deve ser validado por walk-forward.

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
Backtest das distribuições nesses K concursos
       ↓
Distribuição com mais 13+
```

A similaridade deve utilizar exclusivamente informações disponíveis antes do resultado real.

---

# Concentração histórica de Top1

A regra de “concentrar Top1 nas primeiras 9 posições” deve ser substituída por experimentos verificáveis.

## Alternativa 1 — Top1 por posição

Calcular:

```text
Top1_hit[J01]
Top1_hit[J02]
...
Top1_hit[J14]
```

## Alternativa 2 — testar diferentes cortes

Comparar:

```text
Top1 nas primeiras 5
Top1 nas primeiras 7
Top1 nas primeiras 9
Top1 nas primeiras 10
Top1 nas primeiras 12
```

Sem assumir previamente que 9 é o melhor corte.

## Alternativa 3 — runs

Calcular por concurso:

```text
longest_top1_run
number_of_top1_runs
```

## Alternativa 4 — fragmentação

Calcular:

```text
number_of_transitions
fragmentation_score
```

Exemplo:

```text
T1 T1 T1 T1 T2 T2 T1
```

tem menos fragmentação que:

```text
T1 T2 T1 T2 T1 T2 T1
```

Runs e fragmentação devem ser usadas apenas como critério de otimização se demonstrarem associação robusta com 13+ fora da amostra. Caso contrário, podem permanecer somente como informação de auditoria.

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
- escolher políticas, distribuições e hiperparâmetros apenas com informação passada.

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

Salvar também resultados concurso a concurso em:

```text
output/backtest.csv
```

Campos sugeridos:

```text
concurso
strategy
distribution_id
hits
hit_14
hit_13
hit_12
double_games
ticket
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

# Calibração das probabilidades

Como os otimizadores probabilísticos dependem diretamente de `p(1)`, `p(X)` e `p(2)`, a calibração deve ser medida por:

```text
Log Loss
Brier Score
Calibration Error
Reliability Diagram
```

Também avaliar faixas como:

```text
Top1 33–40%
Top1 40–45%
Top1 45–50%
Top1 50–60%
Top1 >60%
```

para comparar probabilidade prevista e frequência observada.

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
```

Evoluções desejadas:

```text
distribuição T1/T2/T3
historical_score
position_score
similarity_score
fragmentation_score
longest_top1_run
impacto quantitativo das constraints
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
```

---

# Roadmap resumido

## Concluído

- [x] pipeline único de constraints;
- [x] invariantes 9/5/0;
- [x] políticas gain/uncertainty/margin/ratio;
- [x] `exact` com 2.002 combinações;
- [x] `P(14)`, `P(13)`, `P(>=13)` e `E[acertos]`;
- [x] ganho marginal dos duplos.

## Próximas prioridades

- [ ] `distribution_backtest` Top1/Top2/Top3;
- [ ] permitir duplos T1T3 e T2T3;
- [ ] avaliar secos Top2/Top3;
- [ ] matriz posição × ranking;
- [ ] FullMarkingOptimizer;
- [ ] `historical_exact`;
- [ ] `probability_exact` completo;
- [ ] `hybrid_exact`;
- [ ] distribuição histórica por similaridade/KNN;
- [ ] walk-forward;
- [ ] backtest detalhado 10–14;
- [ ] baseline aleatório;
- [ ] bootstrap e intervalos de confiança;
- [ ] validação de runs e fragmentação;
- [ ] calibração probabilística;
- [ ] otimização do limiar do Palmeiras.

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
Distribuição Top1/Top2/Top3
      +
Constraints
      +
Backtest walk-forward
      +
Otimização
      ↓
PALPITE FINAL
```

A direção do projeto é tornar o sistema progressivamente menos dependente de regras intuitivas e mais orientado por evidência:

> **maximizar, de forma auditável e validada fora da amostra, a capacidade da aposta de atingir 13 ou 14 pontos.**
