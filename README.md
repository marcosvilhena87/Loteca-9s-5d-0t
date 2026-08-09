# Loteca ML — Estratégia 9 Secos / 5 Duplos / 0 Triplos

Projeto de Machine Learning voltado à geração de **um único palpite final da Loteca**, com otimização baseada no histórico de concursos e foco em **maximizar a probabilidade de atingir pelo menos 13 acertos**, respeitando restrições rígidas da estratégia.

## Objetivo

A estratégia utiliza os históricos disponíveis em:

```text
data/concursos_anteriores.csv
```

para estimar probabilidades, ordenar os resultados possíveis de cada partida e construir uma aposta final com:

- **9 secos**;
- **5 duplos**;
- **0 triplos**;
- total de **19 marcações**.

O sistema deve produzir apenas **um palpite final por concurso**.

---

## Probabilidades por partida

Para cada jogo, o modelo deve gerar as probabilidades:

- `p(1)` — vitória do mandante;
- `p(X)` — empate;
- `p(2)` — vitória do visitante.

Exemplo:

```text
p(1) = 0.52
p(X) = 0.28
p(2) = 0.20
```

---

## Ranking probabilístico: Top1 / Top2 / Top3

As três probabilidades são ordenadas da maior para a menor:

- `p(top1)` — maior probabilidade;
- `p(top2)` — segunda maior probabilidade;
- `p(top3)` — menor probabilidade.

### Critério de desempate

Quando duas ou mais probabilidades forem iguais, usar a seguinte prioridade:

```text
1 > 2 > X
```

Exemplo:

```text
p(1) = 0.40
p(X) = 0.20
p(2) = 0.40
```

Ranking:

```text
Top1 = 1
Top2 = 2
Top3 = X
```

---

## One-Hot Encoding do resultado real

O resultado real de cada partida pode ser representado em relação ao ranking probabilístico por:

- `top1_hit`;
- `top2_hit`;
- `top3_hit`.

Exatamente uma dessas variáveis recebe valor `1`.

Exemplo:

Se o resultado real correspondeu ao `Top2`:

```text
top1_hit = 0
top2_hit = 1
top3_hit = 0
```

Essa representação permite medir historicamente quantas vezes o resultado real apareceu em cada posição do ranking probabilístico.

---

# Hard Constraints

As restrições abaixo são obrigatórias e não podem ser violadas pela otimização.

## 1. Estrutura fixa da aposta

Gerar exatamente:

```text
9 secos
5 duplos
0 triplos
```

Como cada seco utiliza 1 marcação e cada duplo utiliza 2:

```text
9 × 1 + 5 × 2 = 19 marcações
```

## 2. FLAMENGO/RJ

Quando o **FLAMENGO/RJ** participar do concurso, a aposta deve obrigatoriamente conter o resultado correspondente à sua vitória.

Isso significa:

- Flamengo como mandante → incluir `1`;
- Flamengo como visitante → incluir `2`.

A vitória pode estar presente em um seco ou em um duplo, desde que esteja coberta pela aposta.

---

# Soft Constraints

As regras abaixo devem influenciar a otimização, mas podem ser flexibilizadas quando entrarem em conflito com critérios de maior importância.

## 1. Concentração de Top1

Favorecer ordenações que:

- antecipem resultados `Top1`;
- concentrem `Top1` principalmente nas **9 primeiras posições**;
- produzam sequências longas de Top1;
- reduzam fragmentação entre Top1, Top2 e Top3.

Entre soluções de qualidade semelhante, deve ser preferida aquela cuja estrutura apresente maior concentração dos resultados mais prováveis.

## 2. PALMEIRAS/SP

Favorecer soluções que excluam a vitória do **PALMEIRAS/SP**, priorizando:

- empate; ou
- derrota do Palmeiras.

Essa preferência somente deve ser aplicada quando não comprometer significativamente a qualidade global da aposta.

Na implementação atual, a substituição de um seco vencedor do Palmeiras somente ocorre quando a perda de probabilidade para a melhor alternativa é de, no máximo, `0.03`.

Esse limiar é uma **Soft Constraint** e deverá futuramente ser validado e, idealmente, otimizado por backtest.

---

# Estratégia atual de geração da aposta

O fluxo atual é:

```text
Dados históricos
      ↓
Leitura e normalização
      ↓
p(1), p(X), p(2)
      ↓
Ranking Top1 / Top2 / Top3
      ↓
Backtest de políticas de alocação dos 5 duplos
      ↓
Seleção da melhor política histórica
      ↓
Construção de 9 secos + 5 duplos
      ↓
Aplicação das constraints
      ↓
Cálculo de P(>=13)
      ↓
Palpite final
```

As estratégias de alocação atualmente comparadas são:

```text
gain
uncertainty
margin
ratio
exact
```

Cada política atribui um score às 14 partidas e escolhe os 5 jogos de maior prioridade para receber duplo.

`exact` avalia exaustivamente as 2.002 combinações de cinco duplos e maximiza
`P(>=13)`, usando `P(14)`, acertos esperados e concentração nos primeiros jogos
como desempates determinísticos. As demais estratégias permanecem como benchmarks.

### Critério atual de seleção da estratégia

A política vencedora é escolhida priorizando:

1. maior número histórico de concursos com **13 ou 14 acertos**;
2. maior número de **14 acertos**;
3. maior total agregado de acertos;
4. ordem fixa das políticas como último desempate.

---

# Estado atual do backtest

Com a base histórica atual, foram avaliados **445 concursos**.

Exemplo de execução:

```text
[TRAIN] 445 concursos; Top hits: [0.517817, 0.265329, 0.216854]
[BACKTEST] {
  'gain':        {'14': 0, '13': 7, 'hits': 3900},
  'uncertainty': {'14': 0, '13': 8, 'hits': 3888},
  'margin':      {'14': 0, '13': 7, 'hits': 3892},
  'ratio':       {'14': 0, '13': 8, 'hits': 3886}
}
```

Na amostra atual:

```text
Top1: 51.7817%
Top2: 26.5329%
Top3: 21.6854%
```

Esses valores descrevem a posição ocupada pelo resultado real no ranking probabilístico histórico.

> Importante: os números acima representam o estado atual da base e podem mudar quando novos concursos forem adicionados.

---

# Limitação importante da implementação atual

Atualmente, o backtest histórico avalia as políticas por meio da construção básica do ticket, enquanto as regras específicas de Flamengo e Palmeiras são aplicadas posteriormente na geração do palpite final.

Isso significa que o backtest e a execução final ainda não percorrem exatamente o mesmo pipeline.

Uma das prioridades do projeto é criar uma função única de construção de ticket com constraints, utilizada tanto no backtest quanto na previsão futura.

Fluxo desejado:

```text
                 ┌──────────── Backtest histórico
                 │
build_ticket() ──┤
+ constraints    │
                 └──────────── Próximo concurso
```

Assim, toda estratégia será avaliada historicamente exatamente da mesma forma como será utilizada no concurso futuro.

---

# Otimização exata dos cinco duplos — melhoria prioritária

A estratégia atual utiliza heurísticas para selecionar os cinco jogos que receberão duplo.

Entretanto, escolher exatamente 5 partidas entre 14 produz apenas:

```text
C(14,5) = 2.002 combinações
```

Esse espaço de busca é suficientemente pequeno para permitir **otimização exata**.

A evolução prioritária consiste em avaliar todas as 2.002 combinações possíveis de cinco duplos.

Para cada candidato:

```text
9 secos + 5 duplos
        ↓
probabilidade de cobertura por jogo
        ↓
P(14)
P(13)
P(>=13)
E[acertos]
        ↓
Hard Constraints
        ↓
Soft Constraints
        ↓
score final
```

O objetivo principal será:

```text
argmax P(>=13)
```

com critérios secundários para desempate.

Esse otimizador exato deverá ser comparado diretamente com as políticas heurísticas atuais.

---

# Probabilidade de 13 e 14 acertos

A implementação atual calcula a probabilidade de pelo menos 13 acertos utilizando uma distribuição **Poisson-binomial**, assumindo independência entre os jogos.

Para cada partida é calculada a probabilidade coberta pela aposta:

### Seco

```text
q(i) = P(resultado escolhido)
```

### Duplo

```text
q(i) = P(resultado A) + P(resultado B)
```

A evolução da telemetria deverá exibir separadamente:

```text
P(14)
P(13)
P(>=13)
E[acertos]
```

Exemplo desejado:

```text
[METRIC]
  P(14):       0.012345%
  P(13):       0.437836%
  P(>=13):     0.450181%
  E[acertos]:  10.42
```

A hipótese de independência deverá permanecer explicitamente documentada enquanto for utilizada.

---

# Ganho marginal dos duplos

Para tornar a decisão auditável, cada candidato a duplo deverá registrar o ganho de cobertura proporcionado pela segunda marcação.

Exemplo:

```text
[JOGO 07]
Top1: 1 = 0.358197
Top2: 2 = 0.351464

Cobertura seco:  35.8197%
Cobertura duplo: 70.9661%
Ganho marginal: +35.1464 p.p.
Ranking para duplo: 1/14
```

Isso permite entender claramente por que uma partida recebeu um dos cinco duplos disponíveis.

---

# Roadmap de aprimoramentos

## Prioridade 1 — Consistência do pipeline

- [x] aplicar as mesmas Hard e Soft Constraints no backtest e na previsão;
- [x] criar uma função única para construção do ticket final;
- [x] garantir invariantes: 9 secos, 5 duplos, 0 triplos e 19 marcações.

## Prioridade 2 — Otimizador exato

- [x] gerar as 2.002 combinações possíveis de cinco duplos;
- [x] calcular `P(14)`, `P(13)` e `P(>=13)` para cada combinação;
- [x] selecionar diretamente a combinação com melhor objetivo;
- [x] comparar o resultado com `gain`, `uncertainty`, `margin` e `ratio`.

## Prioridade 3 — Backtest completo

- [ ] registrar distribuição de 10, 11, 12, 13 e 14 acertos;
- [ ] calcular média, mediana e desvio-padrão dos acertos;
- [ ] salvar os resultados detalhados em `output/backtest.csv`;
- [ ] permitir investigar concurso a concurso onde duas políticas divergiram.

## Prioridade 4 — Walk-forward validation

Evitar avaliar uma estratégia utilizando informação futura.

Fluxo desejado:

```text
Concursos 1..N     → treinamento
Concurso N+1       → teste
Concursos 1..N+1   → treinamento
Concurso N+2       → teste
...
```

- [ ] implementar backtest walk-forward;
- [ ] comparar desempenho in-sample e out-of-sample;
- [ ] usar resultados fora da amostra para selecionar políticas e hiperparâmetros.

## Prioridade 5 — Benchmarks

Comparar o otimizador contra estratégias-base:

```text
5 maiores p(Top2)
5 menores margens Top1-Top2
5 maiores incertezas
5 maiores razões Top2/Top1
5 duplos aleatórios
otimizador exato P(>=13)
```

- [ ] criar baseline aleatório com múltiplas sementes;
- [ ] medir ganho real sobre o acaso;
- [ ] evitar atribuir valor a diferenças pequenas sem suporte estatístico.

## Prioridade 6 — Incerteza estatística

- [ ] bootstrap com 1.000 ou mais reamostragens;
- [ ] intervalos de confiança para `P(13+)` histórico;
- [ ] comparação entre políticas considerando variabilidade;
- [ ] evitar concluir superioridade com diferenças de apenas um concurso.

## Prioridade 7 — Calibração das probabilidades

Antes de aumentar a complexidade do modelo, medir a qualidade probabilística com:

```text
Log Loss
Brier Score
Calibration Error
Reliability Diagram
```

Também avaliar faixas de confiança:

```text
Top1 33–40%
Top1 40–45%
Top1 45–50%
Top1 50–60%
Top1 >60%
```

para verificar se as probabilidades previstas correspondem às frequências observadas.

## Prioridade 8 — Análises históricas adicionais

- [ ] Top1/Top2/Top3 por posição do jogo (`Jogo 01` a `Jogo 14`);
- [ ] desempenho por faixa probabilística;
- [ ] análise da frequência de erros simultâneos;
- [ ] análise das distribuições de resultados dentro de cada concurso;
- [ ] validação de qualquer padrão sempre fora da amostra.

## Prioridade 9 — Soft Constraint do Palmeiras

O limite atual de `0.03` não deve permanecer como valor arbitrário sem validação.

Testar historicamente valores como:

```text
0.00
0.01
0.02
0.03
0.04
0.05
...
0.15
```

- [ ] transformar o limite em parâmetro configurável;
- [ ] avaliar seu impacto em walk-forward;
- [ ] aplicar a preferência somente quando houver evidência histórica favorável.

## Prioridade 10 — Testes automatizados

Adicionar testes para garantir:

- [ ] exatamente 14 partidas por concurso;
- [ ] exatamente 9 secos;
- [ ] exatamente 5 duplos;
- [ ] exatamente 19 marcações;
- [ ] ausência de triplos;
- [ ] Flamengo sempre coberto;
- [ ] critério de empate `1 > 2 > X`;
- [ ] probabilidades normalizadas;
- [ ] constraints não alteram a estrutura `9/5/0`;
- [ ] cálculo de `P(13)` e `P(14)` validado contra casos conhecidos.

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

# Arquivos de dados

## `data/concursos_anteriores.csv`

Base histórica utilizada para:

- treinamento;
- validação;
- backtesting;
- cálculo das frequências de Top1, Top2 e Top3;
- avaliação das políticas de alocação dos cinco duplos;
- escolha da estratégia final.

## `data/proximo_concurso.csv`

Contém os 14 jogos do concurso que será utilizado para gerar o próximo palpite.

---

# Formato dos CSVs

Os arquivos utilizam:

```text
Delimitador: ;
Separador decimal: ,
```

A leitura aceita UTF-8 e também bases legadas em CP1252/Latin-1.

As probabilidades lidas são normalizadas para que:

```text
p(1) + p(X) + p(2) = 1
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

Na estratégia principal deste projeto:

```text
Triplos = 0
```

---

# Telemetria e auditoria no terminal

O programa deve fornecer informações suficientes para compreender como cada decisão foi tomada.

A saída atual permite visualizar:

- probabilidades `p(1)`, `p(X)` e `p(2)`;
- classificação Top1 / Top2 / Top3;
- escolha entre seco e duplo;
- política histórica selecionada;
- Hard Constraint do Flamengo;
- Soft Constraint do Palmeiras;
- probabilidade estimada de pelo menos 13 acertos;
- palpite final.

Exemplo:

```text
[INFO] Concurso: 1263
[INFO] Estratégia: 9 secos / 5 duplos / 0 triplos
[OPT] Política histórica selecionada: uncertainty

[JOGO 07] VILA NOVA-GO x FORTALEZA-CE
  p(1): 0.358197  p(X): 0.290340  p(2): 0.351464
  Top1/Top2/Top3: 1/2/X
  Escolha: DUPLO 12

[CONSTRAINT] PALMEIRAS jogo 9: preferência não aplicada
[CONSTRAINT] FLAMENGO jogo 12: vitória 2 coberta
[METRIC] Probabilidade estimada de >=13: 0.450181%
[FINAL] 9 secos / 5 duplos / 0 triplos — 19 marcações
```

A evolução planejada da telemetria inclui:

```text
P(14)
P(13)
P(>=13)
Expected hits
Ganho marginal de cada duplo
Ranking dos candidatos a duplo
Impacto quantitativo das constraints
```

---

# `output/predictions.csv`

O arquivo final registra informações suficientes para reconstruir e auditar o palpite:

```text
concurso
jogo
mandante
visitante
p_1
p_x
p_2
top1_result
top1_prob
top2_result
top2_prob
top3_result
top3_prob
tipo_aposta
palpite
```

A evolução planejada poderá acrescentar:

```text
covered_probability
double_gain
double_rank
resultado_real
top1_hit
top2_hit
top3_hit
```

---

# Execução

O pipeline atual não requer dependências externas.

Para treinar, executar o backtest das políticas e gerar o próximo palpite:

```bash
python main.py
```

Também é possível informar caminhos alternativos:

```bash
python main.py \
  --history data/concursos_anteriores.csv \
  --next data/proximo_concurso.csv \
  --model models/model.json \
  --output output/predictions.csv
```

No PowerShell/Windows, a execução padrão é:

```powershell
python main.py
```

---

# Testes automatizados

Executar:

```bash
python -m unittest discover -v
```

Os testes devem evoluir junto com o otimizador para cobrir as invariantes estruturais, constraints e métricas probabilísticas.

---

# Critério de sucesso

O objetivo do projeto **não é simplesmente maximizar a acurácia individual dos palpites secos**.

A métrica principal deve considerar o desempenho da **aposta completa de 19 marcações**.

Ordem conceitual de importância:

```text
14 acertos
   ↓
13 acertos
   ↓
12 acertos / estabilidade
   ↓
média de acertos
```

A escolha entre duas estratégias deve considerar o desempenho do conjunto completo da aposta e não apenas a taxa de acerto do `Top1` isoladamente.

---

# Princípio geral

O modelo fornece as probabilidades.

O ranking Top1 / Top2 / Top3 organiza essas probabilidades.

O histórico informa quais decisões funcionaram melhor fora da amostra.

As Hard Constraints definem o espaço permitido.

As Soft Constraints ajudam a desempatar soluções semelhantes.

O otimizador seleciona **um único jogo final de 9 secos e 5 duplos**.

```text
Probabilidade
     +
Histórico
     +
Constraints
     +
Backtest
     +
Otimização exata
     ↓
PALPITE FINAL
```

## Direção do projeto

A prioridade é tornar a estratégia progressivamente menos dependente de heurísticas e mais diretamente alinhada ao objetivo final:

> **maximizar, de forma auditável e validada fora da amostra, a capacidade da aposta de atingir 13 ou 14 pontos.**
