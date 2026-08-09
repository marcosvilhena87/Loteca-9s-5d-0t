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

# Princípio central — preservar o Top1 até existir evidência melhor

O `p(Top1)` é o baseline individual mais forte do projeto.

A pesquisa histórica não deve procurar simplesmente uma métrica diferente. Uma nova métrica só deve substituir ou reordenar `p(Top1)` se demonstrar **informação incremental fora da amostra**.

Critérios mínimos:

```text
1. superar p(Top1) em walk-forward;
2. vencer quando sua ordenação discorda de p(Top1);
3. melhorar P13+ e/ou P12+ do ticket;
4. apresentar estabilidade temporal;
5. não usar informação futura.
```

A evidência atual indica que as correções históricas testadas ainda não cumprem essa regra.

---

# Estado atual das correções do Top1

Foram testadas:

```text
top1_residual
top1_lift
top1_reliability
p(top1_meta)
```

Resultado walk-forward atual:

```text
[DISAGREEMENT] top1_residual
3234 casos | baseline 802 x histórico 750 | neutros 1682
historical win rate: 48.32%

[DISAGREEMENT] top1_lift
3264 casos | baseline 801 x histórico 755 | neutros 1708
historical win rate: 48.52%

[DISAGREEMENT] top1_reliability
3323 casos | baseline 821 x histórico 746 | neutros 1756
historical win rate: 47.61%

[TOP1-META]
Brier baseline: 0.233977
Brier meta:     0.240629

[DISAGREEMENT] p_top1_meta
4107 casos | baseline 1140 x meta 896 | neutros 2071
meta win rate: 44.01%
```

Conclusão atual:

> **nenhuma das correções testadas superou `p(Top1)` fora da amostra.**

Essas métricas permanecem como benchmarks e telemetria. Não devem alterar o ticket final enquanto não demonstrarem ganho incremental.

---

# p(top1_meta) — status experimental congelado

O meta-modelo usa regressão logística para estimar:

```text
p(top1_meta) = P(Top1_hit | contexto)
```

Features atuais:

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

Como primeira revisão técnica, testar uma versão reduzida para diminuir redundâncias matemáticas:

```text
p_top1
margin_top1_top2
entropy
top1_is_X
top1_is_2
```

Também validar uma implementação de referência com **refit completo em todo o histórico disponível a cada passo walk-forward**, em vez de depender apenas da atualização online.

Mesmo que essa revisão seja feita, `p(top1_meta)` permanece congelado até superar o baseline em Brier e disagreement.

---

# Disagreement por intensidade

Para qualquer score candidato, medir:

```text
delta_score = |score_candidato - p(top1)|
```

Faixas:

```text
< 0.02
0.02–0.05
0.05–0.10
>= 0.10
```

Objetivo:

> descobrir se alguma métrica só acrescenta valor quando produz uma discordância forte.

---

# Disagreement por faixa de p(Top1)

Separar por nível de confiança do favorito:

```text
33–40%
40–45%
45–50%
50–60%
60%+
```

Uma métrica pode perder globalmente e ainda acrescentar sinal em um regime específico. Qualquer uso condicional deve ser validado fora da amostra.

---

# Nova prioridade principal — recuperar erros do Top1

Como `p(Top1)` continua difícil de superar, a principal oportunidade histórica passa a ser:

> **quando o Top1 estiver errado, qual segunda marcação tem maior capacidade de recuperar esse erro?**

Isso está diretamente alinhado à estrutura de **5 duplos**.

Hoje o desenho dominante é:

```text
Seco  = Top1
Duplo = Top1 + Top2
```

A próxima investigação deve comparar:

```text
Top1 + Top2
vs
Top1 + Top3
```

mantendo Top1 coberto.

---

# Separação arquitetural — DoubleAllocator × SecondMarkSelector

A estratégia deve ser dividida em dois problemas independentes.

## DoubleAllocator

Decide **quais cinco jogos recebem duplo**.

Políticas atuais:

```text
gain
uncertainty
margin
ratio
exact
hist_top1
hist_top2
```

## SecondMarkSelector

Decide **qual resultado acompanha o Top1 em cada duplo**.

Candidatos:

```text
top2_baseline
recovery
hybrid_double_value
```

Fluxo desejado:

```text
14 jogos
   ↓
DoubleAllocator
   ↓
5 jogos com duplo
   ↓
SecondMarkSelector
   ↓
T1T2 ou T1T3 em cada duplo
   ↓
constraints
   ↓
ticket final
```

Essa separação permite descobrir se o ganho vem da escolha dos jogos duplos ou da escolha da segunda marcação.

---

# Error Recovery Score

Criar scores históricos condicionados ao erro do Top1:

```text
recovery_top2 = P(Top2_hit | Top1_miss, contexto)
recovery_top3 = P(Top3_hit | Top1_miss, contexto)
```

Como Top2 e Top3 são os únicos resultados possíveis quando Top1 falha:

```text
recovery_top2 + recovery_top3 = 1
```

por contexto, salvo suavização/estimativa.

Contexto candidato:

```text
p_top1
p_top2
p_top3
margin_top1_top2
margin_top2_top3
entropy
top1_is_1/X/2
perfil probabilístico do concurso
```

A primeira versão deve ser simples e regularizada para evitar overfitting.

Objetivo:

> escolher a melhor proteção do Top1, e não substituir o Top1.

---

# Second-Mark Disagreement Test

Criar um disagreement específico da segunda marcação.

Baseline:

```text
p(Top2) > p(Top3)
→ escolher Top2
```

Histórico:

```text
recovery_top3 > recovery_top2
→ escolher Top3
```

Avaliar somente casos em que:

```text
Top1 errou
```

E, entre esses casos, focar nas situações em que baseline e recovery discordaram.

Saída desejada:

```text
[SECOND-MARK DISAGREEMENT]
casos: N
Top2 baseline wins: X
recovery wins: Y
neutros: Z
recovery win rate: ...
```

Critério mínimo para promoção:

```text
recovery win rate > 50%
```

mas a decisão final deve considerar também IC95%, estabilidade temporal e impacto no ticket.

---

# Second-Mark Disagreement por intensidade

Medir a força da preferência histórica:

```text
recovery_advantage = |recovery_top2 - recovery_top3|
```

Faixas sugeridas:

```text
< 0.02
0.02–0.05
0.05–0.10
>= 0.10
```

Possível regra futura:

```text
usar T1T2 normalmente
usar T1T3 apenas quando recovery_top3 - recovery_top2 >= threshold
```

O `threshold` deve ser escolhido por walk-forward.

---

# Second-Mark Disagreement por faixa de p(Top1)

Também segmentar o recovery por confiança do Top1:

```text
33–40%
40–45%
45–50%
50–60%
60%+
```

Isso pode revelar que a escolha histórica de Top3 só agrega valor em partidas muito equilibradas ou em regimes específicos.

---

# Double Value Score

Combinar probabilidade atual e recuperação histórica:

```text
score_T2 = α × p(Top2) + (1-α) × recovery_top2
score_T3 = α × p(Top3) + (1-α) × recovery_top3
```

Testar inicialmente:

```text
α = 0.00
α = 0.25
α = 0.50
α = 0.75
α = 1.00
```

Interpretação:

```text
α = 1 → probabilidade pura
α = 0 → histórico puro
```

O valor de `α` deve ser escolhido por walk-forward.

Benchmarks obrigatórios:

```text
Top1+Top2 baseline
Top1+melhor_recovery
Top1+double_value_score
```

Uma segunda marcação histórica só deve ser promovida se melhorar P13+/P12+ fora da amostra.

---

# Backtest matricial — Allocator × SecondMarkSelector

Comparar sistematicamente:

```text
                     Top2 baseline   Recovery   DoubleValue
gain                       ...          ...          ...
uncertainty                ...          ...          ...
margin                     ...          ...          ...
ratio                      ...          ...          ...
exact                      ...          ...          ...
```

Cada combinação deve registrar:

```text
14
13
12
11
10
<=9
P13+ empírico
P12+ empírico
média
mediana
desvio-padrão
```

Esse backtest deve responder separadamente:

```text
qual allocator funciona melhor?
qual second-mark selector funciona melhor?
qual combinação dos dois produz mais 13+?
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

As constraints devem continuar valendo em treino, backtest e previsão final.

---

# Soft Constraint — Palmeiras

Favorecer, quando o custo probabilístico for pequeno, soluções que excluam a vitória do **PALMEIRAS/SP**.

Limiar atual:

```text
0.03
```

Esse valor é experimental e deve ser otimizado apenas com validação walk-forward.

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

Protege com duplos as posições onde Top1 foi historicamente menos confiável.

## hist_top2

Prioriza posições com maior frequência histórica de Top2.

## exact

Avalia:

```text
C(14,5) = 2.002
```

alocações dos cinco duplos e maximiza principalmente:

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

Critério atual de seleção:

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

---

# P13+ e P12+ empíricos

```text
P13+ empírico = concursos com 13 ou 14 / concursos testados
P12+ empírico = concursos com 12, 13 ou 14 / concursos testados
```

Essas medidas são diferentes das probabilidades teóricas Poisson-binomial do ticket.

Resultados concurso a concurso:

```text
output/backtest.csv
```

---

# Bootstrap e incerteza estatística

Implementar:

```text
bootstrap >= 1.000 reamostragens
IC95% de P13+
IC95% de P12+
IC95% das diferenças entre estratégias
IC95% do Top1 Disagreement
IC95% do Second-Mark Disagreement
```

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

O ECE baixo estabelece um baseline probabilístico forte.

Calibração aplicada futura:

```text
exact_raw
vs
exact_calibrated
```

Só manter calibração aplicada se melhorar walk-forward.

---

# Distribution Backtest

Depois de entender melhor o valor da segunda marcação, ampliar o espaço das 19 marcações.

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

# FullMarkingOptimizer

O FullMarkingOptimizer só deve entrar depois de validar onde existe sinal incremental.

Primeira expansão:

```text
Seco:  T1
Duplo: T1T2 | T1T3
```

Somente depois, se houver evidência forte, abrir:

```text
Seco:  T1 | T2 | T3
Duplo: T1T2 | T1T3 | T2T3
```

O histórico só recebe peso se demonstrar ganho incremental fora da amostra.

---

# Similaridade histórica / KNN

Criar futuramente `similarity_knn` usando apenas informação pré-jogo.

Features candidatas:

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

O KNN pode ser útil principalmente para estimar recuperação Top2/Top3 em concursos probabilisticamente semelhantes.

---

# Estabilidade temporal

Medir desempenho em terços do histórico e janelas móveis.

Registrar:

```text
P13+ por período
P12+ por período
Top1 accuracy
Top1 disagreement win rate
Second-Mark disagreement win rate
recovery_top3 usage rate
média por período
pior janela
melhor janela
```

---

# Telemetria desejada

Por jogo:

```text
p_top1
p_top2
p_top3
margin_top1_top2
margin_top2_top3
entropy
p_top1_meta
top1_meta_delta
recovery_top2
recovery_top3
recovery_advantage
second_mark_baseline
second_mark_recovery
second_mark_final
second_mark_disagreement_flag
double_value_top2
double_value_top3
double_value_alpha
```

Agregados:

```text
Top1 baseline Brier
Top1 meta Brier
Top1 disagreement win rate
Second-Mark disagreement win rate
T1T2 usage rate
T1T3 usage rate
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
- [x] Brier, Log Loss e ECE;
- [x] matriz posição × ranking;
- [x] `hist_top1` e `hist_top2`;
- [x] walk-forward sem vazamento temporal;
- [x] backtest 10–14;
- [x] `P13+` e `P12+` empíricos;
- [x] `output/backtest.csv`;
- [x] Disagreement Test contra `p(Top1)`;
- [x] `top1_residual`;
- [x] `top1_lift`;
- [x] `top1_reliability`;
- [x] `p(top1_meta)`;
- [x] disagreement por intensidade;
- [x] disagreement por faixa de `p(Top1)`;
- [x] evidência de que as quatro correções atuais não superam `p(Top1)`.

## Próximas prioridades — ordem prática

1. [ ] validar `p(top1_meta)` com refit completo e features reduzidas;
2. [ ] congelar formalmente como benchmark correções Top1 que permaneçam <=50%;
3. [ ] implementar `error_recovery_score` para Top2 e Top3;
4. [ ] implementar `SecondMarkSelector` separado do `DoubleAllocator`;
5. [ ] comparar `T1T2` vs `T1T3` em walk-forward;
6. [ ] implementar Second-Mark Disagreement Test;
7. [ ] segmentar Second-Mark Disagreement por intensidade;
8. [ ] segmentar Second-Mark Disagreement por faixa de `p(Top1)`;
9. [ ] bootstrap + IC95% da segunda marcação;
10. [ ] implementar `double_value_score`;
11. [ ] otimizar `alpha` por walk-forward;
12. [ ] otimizar threshold para troca `T2 → T3`;
13. [ ] implementar backtest matricial `Allocator × SecondMarkSelector`;
14. [ ] medir impacto de recovery sobre P13+/P12+;
15. [ ] medir estabilidade temporal;
16. [ ] implementar `distribution_backtest` Top1/Top2/Top3;
17. [ ] implementar FullMarkingOptimizer inicialmente com T1T2/T1T3;
18. [ ] avaliar T2T3 e secos Top2/Top3 somente se houver evidência;
19. [ ] implementar KNN por similaridade;
20. [ ] implementar calibração aplicada;
21. [ ] testar `historical_13plus_score` como métrica complementar;
22. [ ] adicionar baseline aleatório;
23. [ ] remover/substituir desempate posicional arbitrário do `exact`;
24. [ ] otimizar o limiar do Palmeiras.

---

# Critério de sucesso

A régua atual é:

> **preservar o que `p(Top1)` já faz bem e usar o histórico principalmente para melhorar as cinco marcações adicionais.**

Para o Top1:

```text
superar p(Top1) fora da amostra
↓
vencer nos casos de discordância
↓
melhorar Brier
↓
resistir a bootstrap
```

Para a segunda marcação:

```text
recuperar mais erros do Top1 que Top2 puro
↓
vencer Second-Mark Disagreement
↓
melhorar P13+ / P12+
↓
manter estabilidade temporal
↓
resistir a bootstrap
```

A unidade final de avaliação continua sendo o ticket completo de 19 marcações.

---

# Princípio geral

```text
p(Top1) baseline preservado
      +
DoubleAllocator
      +
SecondMarkSelector
      +
Error Recovery Score
      +
T1T2 vs T1T3
      +
Validação walk-forward
      +
Incerteza estatística
      +
Distribution Backtest
      +
Constraints
      +
Otimização
      ↓
PALPITE FINAL
```

> **O histórico não precisa vencer o Top1 para ser útil: pode gerar valor escolhendo melhor qual resultado deve acompanhá-lo nos cinco duplos.**
