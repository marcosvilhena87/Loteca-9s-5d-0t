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

A evidência atual indica que pequenas correções históricas do Top1 ainda não conseguiram cumprir essa regra.

---

# Estado atual do Disagreement Test

Foram testadas três correções históricas de confiabilidade do Top1:

```text
top1_residual
top1_lift
top1_reliability
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
```

Conclusão atual:

> `top1_residual`, `top1_lift` e `top1_reliability` **não superaram `p(Top1)`** nos pares informativos de discordância.

Essas métricas permanecem como telemetria/benchmark e **não devem alterar o ticket final** enquanto continuarem abaixo do baseline.

---

# Próxima tentativa de superar p(Top1) — p(top1_meta)

Criar um modelo secundário cujo alvo seja:

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
p(top1_meta) = P(Top1 acertar | contexto)
```

O objetivo é aprender **quando confiar mais ou menos no próprio Top1**, sem ignorar a probabilidade original.

Comparação obrigatória:

```text
p(top1) vs p(top1_meta)
```

sempre em walk-forward.

---

# Disagreement por intensidade

Além de contar vitórias, medir a intensidade da correção:

```text
delta_score = |score_candidato - p(top1)|
```

Faixas sugeridas:

```text
< 0.02
0.02–0.05
0.05–0.10
> 0.10
```

Objetivo:

> descobrir se o histórico só acrescenta valor quando produz uma discordância forte.

Saída desejada:

```text
[DISAGREEMENT-STRENGTH]
metric: p_top1_meta
faixa: >0.05
casos: N
baseline_wins: X
meta_wins: Y
meta_win_rate: ...
```

---

# Disagreement por faixa de p(Top1)

Separar o teste por nível de confiança do favorito:

```text
33–40%
40–45%
45–50%
50–60%
60%+
```

Uma métrica pode perder globalmente e ainda acrescentar sinal em um regime específico, por exemplo apenas em partidas muito equilibradas.

Qualquer uso condicional deve ser validado fora da amostra.

---

# Nova linha prioritária — recuperar erros do Top1

Como `p(Top1)` continua difícil de superar, a principal oportunidade histórica passa a ser outra:

> **quando o Top1 estiver errado, qual segunda marcação tem maior capacidade de recuperar o erro?**

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

por contexto.

---

# Error Recovery Score

Criar scores históricos condicionados ao erro do Top1:

```text
recovery_top2 = P(Top2_hit | Top1_miss, contexto)
recovery_top3 = P(Top3_hit | Top1_miss, contexto)
```

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

O objetivo não é substituir o Top1, mas escolher a **melhor proteção** para ele.

---

# Second-Mark Disagreement Test

Criar um disagreement específico da segunda marcação.

Exemplo:

```text
p(Top2) prefere Top2
recovery_score prefere Top3
```

Pergunta de avaliação:

> quando Top1 falhou e os dois critérios discordaram, qual segunda marcação acertou mais?

Saída desejada:

```text
[SECOND-MARK DISAGREEMENT]
casos: N
Top2 baseline wins: X
recovery wins: Y
neutros: Z
recovery win rate: ...
```

Essa comparação deve usar apenas concursos anteriores em cada passo walk-forward.

---

# Double Value Score

Combinar probabilidade atual e recuperação histórica:

```text
score_T2 = α × p(Top2) + (1-α) × recovery_top2
score_T3 = α × p(Top3) + (1-α) × recovery_top3
```

O valor de `α` deve ser aprendido por walk-forward.

Benchmark obrigatório:

```text
Top1+Top2 baseline
vs
Top1+melhor_recovery
vs
Top1+double_value_score
```

Uma segunda marcação histórica só deve ser promovida se melhorar P13+/P12+ fora da amostra.

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
IC95% do Disagreement Test
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

Espaço futuro:

```text
Seco:  T1 | T2 | T3
Duplo: T1T2 | T1T3 | T2T3
```

Fluxo:

```text
14 jogos
   ↓
opções de marcação
   ↓
Hard Constraints
   ↓
9 secos + 5 duplos
   ↓
score probabilístico + histórico validado
   ↓
melhor ticket
```

Comparar:

```text
probability_exact
historical_exact
hybrid_exact
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

O KNN pode ser útil tanto para confiabilidade do Top1 quanto para recuperação Top2/Top3.

---

# Estabilidade temporal

Medir desempenho em terços do histórico e janelas móveis.

Registrar:

```text
P13+ por período
P12+ por período
Top1 accuracy
Disagreement win rate
Second-Mark win rate
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
entropy
p_top1_meta
top1_disagreement_strength
recovery_top2
recovery_top3
second_mark_baseline
second_mark_recovery
second_mark_disagreement_flag
double_value_top2
double_value_top3
```

Agregados:

```text
Top1 baseline accuracy
Top1 meta accuracy
Top1 disagreement win rate
Second-Mark disagreement win rate
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
- [x] evidência de que essas três métricas não superam atualmente o baseline.

## Próximas prioridades — ordem prática

1. [ ] implementar `p(top1_meta)`;
2. [ ] implementar disagreement por intensidade;
3. [ ] implementar disagreement por faixa de `p(Top1)`;
4. [ ] bootstrap + IC95% do disagreement;
5. [ ] congelar como benchmark métricas históricas que permaneçam <=50%;
6. [ ] implementar `error_recovery_score` para Top2 e Top3;
7. [ ] comparar `T1T2` vs `T1T3` em walk-forward;
8. [ ] implementar Second-Mark Disagreement Test;
9. [ ] implementar `double_value_score`;
10. [ ] medir impacto de `recovery_score` sobre P13+/P12+;
11. [ ] medir estabilidade temporal;
12. [ ] implementar `distribution_backtest` Top1/Top2/Top3;
13. [ ] avaliar secos Top2/Top3 somente se houver evidência;
14. [ ] implementar FullMarkingOptimizer;
15. [ ] implementar KNN por similaridade;
16. [ ] implementar calibração aplicada;
17. [ ] testar `historical_13plus_score` como métrica complementar;
18. [ ] adicionar baseline aleatório;
19. [ ] remover/substituir desempate posicional arbitrário do `exact`;
20. [ ] otimizar o limiar do Palmeiras.

---

# Critério de sucesso

A régua atual é:

> **preservar o que `p(Top1)` já faz bem e usar o histórico apenas onde ele demonstrar valor incremental.**

Para o Top1:

```text
superar p(Top1) fora da amostra
↓
vencer nos casos de discordância
↓
resistir a bootstrap
```

Para a segunda marcação:

```text
recuperar mais erros do Top1 que p(Top2) puro
↓
melhorar P13+ / P12+
↓
manter estabilidade temporal
```

A unidade final de avaliação continua sendo o ticket completo de 19 marcações.

---

# Princípio geral

```text
p(Top1) baseline
      +
Meta-confiabilidade validada
      +
Recuperação histórica do erro do Top1
      +
Segunda marcação otimizada
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

> **O histórico não deve substituir o sinal probabilístico por intuição; deve provar onde consegue melhorar aquilo que a probabilidade ainda não resolve.**
