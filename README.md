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

As três probabilidades devem ser ordenadas da maior para a menor:

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

O resultado real de cada partida deve ser representado em relação ao ranking probabilístico por:

- `top1_hit`;
- `top2_hit`;
- `top3_hit`.

Exatamente uma dessas variáveis deve receber valor `1`.

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

# Otimização histórica

A estratégia não deve utilizar uma distribuição fixa de Top1, Top2 e Top3 sem validação histórica.

O sistema deve:

1. considerar as **19 marcações disponíveis**;
2. avaliar todas as distribuições viáveis dessas marcações entre resultados `Top1`, `Top2` e `Top3`;
3. testar o desempenho dessas distribuições no histórico de concursos;
4. selecionar a configuração com melhor desempenho histórico para o objetivo definido.

O critério principal é favorecer a configuração que maximize a capacidade de gerar apostas com **13 ou 14 acertos**.

---

# Soft Constraints

As regras abaixo devem influenciar a otimização, mas podem ser flexibilizadas quando entrarem em conflito com critérios de maior importância.

## 1. Concentração de Top1

Favorecer ordenações que:

- antecipem resultados `Top1`;
- concentrem `Top1` principalmente nas **9 primeiras posições**;
- produzam sequências longas de Top1;
- reduzam fragmentação entre Top1, Top2 e Top3.

Em outras palavras, entre soluções de qualidade semelhante, deve ser preferida aquela cuja estrutura apresente maior concentração dos resultados mais prováveis.

## 2. PALMEIRAS/SP

Favorecer soluções que excluam a vitória do **PALMEIRAS/SP**, priorizando:

- empate; ou
- derrota do Palmeiras.

Essa preferência somente deve ser aplicada quando não comprometer significativamente a qualidade global da aposta.

Portanto, trata-se de uma **Soft Constraint**, e não de uma proibição absoluta da vitória do Palmeiras.

---

# Estratégia de geração da aposta

Para cada concurso, o fluxo esperado é:

```text
Dados históricos
      ↓
Pré-processamento
      ↓
Treinamento / calibração do modelo
      ↓
p(1), p(X), p(2)
      ↓
Ranking Top1 / Top2 / Top3
      ↓
Avaliação histórica das distribuições
      ↓
Aplicação das Hard Constraints
      ↓
Aplicação das Soft Constraints
      ↓
Otimização global
      ↓
9 secos + 5 duplos
      ↓
Palpite final
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
│   ├── preprocess_data.py
│   ├── train_model.py
│   └── predict_results.py
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
- avaliação das diferentes distribuições das 19 marcações;
- escolha da estratégia final.

## `data/proximo_concurso.csv`

Contém os jogos do concurso que será utilizado para gerar o próximo palpite.

---

# Formato dos CSVs

Os arquivos:

```text
data/concursos_anteriores.csv
data/proximo_concurso.csv
```

utilizam:

```text
Delimitador: ;
Separador decimal das odds: ,
```

Exemplo:

```csv
Mandante;Visitante;Odd_1;Odd_X;Odd_2
FLAMENGO/RJ;FLUMINENSE/RJ;1,80;3,40;4,20
```

O código deve realizar a leitura respeitando explicitamente essas configurações.

---

# Formato dos palpites

## Secos

```text
1
X
2
```

## Duplos

Os duplos devem ser apresentados exclusivamente nos formatos:

```text
1X
12
X2
```

## Triplos

Caso sejam utilizados em outros experimentos:

```text
1X2
```

Na estratégia principal deste projeto, entretanto:

```text
Triplos = 0
```

---

# Telemetria e auditoria no terminal

O programa deve fornecer informações suficientes para compreender como cada decisão foi tomada.

A saída deve permitir auditar pelo menos:

- probabilidades `p(1)`, `p(X)` e `p(2)`;
- classificação Top1 / Top2 / Top3;
- escolha entre seco e duplo;
- distribuição das 19 marcações;
- Hard Constraints aplicadas;
- Soft Constraints consideradas;
- critérios utilizados pela otimização;
- palpite final escolhido.

Exemplo de saída esperada:

```text
[INFO] Concurso: 1263
[INFO] Estratégia: 9 secos / 5 duplos / 0 triplos
[INFO] Total de marcações: 19

[JOGO 01] TIME A x TIME B
  p(1): 0.511
  p(X): 0.274
  p(2): 0.215
  Top1: 1
  Top2: X
  Top3: 2
  Escolha: SECO 1

[JOGO 02] TIME C x TIME D
  p(1): 0.380
  p(X): 0.290
  p(2): 0.330
  Top1: 1
  Top2: 2
  Top3: X
  Escolha: DUPLO 12

[OPT] Distribuição selecionada:
  Top1: 14 marcações
  Top2: 5 marcações
  Top3: 0 marcações

[CONSTRAINT] FLAMENGO/RJ: vitória obrigatoriamente coberta
[SOFT] PALMEIRAS/SP: alternativa sem vitória favorecida

[FINAL] 9 secos / 5 duplos / 0 triplos
```

A telemetria deve ser detalhada o suficiente para permitir **logging/debugging da estratégia**, sem transformar a execução em uma caixa-preta.

---

# `output/predictions.csv`

O arquivo final deve registrar, no mínimo, as informações necessárias para reconstruir e auditar o palpite.

Sugestão de campos:

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

Quando o resultado real estiver disponível no histórico, também podem ser registrados:

```text
resultado_real
top1_hit
top2_hit
top3_hit
```

---

# Critério de sucesso

O objetivo do projeto não é simplesmente maximizar a acurácia individual dos palpites secos.

A métrica principal da estratégia deve considerar o desempenho da **aposta completa de 19 marcações**, avaliando historicamente sua capacidade de produzir:

```text
14 acertos
13 acertos
```

A escolha entre duas estratégias deve, portanto, considerar o desempenho do conjunto completo da aposta e não apenas a taxa de acerto do `Top1` isoladamente.

---

# Princípio geral

O modelo fornece as probabilidades.

O ranking Top1 / Top2 / Top3 organiza essas probabilidades.

O histórico informa como distribuir as 19 marcações.

As Hard Constraints definem o espaço permitido.

As Soft Constraints ajudam a desempatar soluções semelhantes.

A otimização global seleciona **um único jogo final de 9 secos e 5 duplos**.

```text
Probabilidade + Histórico + Constraints + Otimização
                        ↓
                PALPITE FINAL
```
