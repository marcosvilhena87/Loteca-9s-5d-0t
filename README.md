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

O `p(Top1)` continua sendo o baseline individual mais forte do projeto.

Uma nova métrica só deve substituir ou reordenar `p(Top1)` se demonstrar **informação incremental fora da amostra**.

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

Essas métricas permanecem como benchmarks/telemetria e **não alteram o ticket final**.

## p(top1_meta) — status congelado

O meta-modelo atual usa regressão logística para estimar:

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

Revisões técnicas opcionais antes de arquivar definitivamente essa linha:

```text
1. reduzir redundâncias entre features;
2. usar Top1=1 como categoria de referência;
3. testar refit completo em todo histórico disponível a cada passo walk-forward.
```

Mesmo com essas revisões, `p(top1_meta)` só poderá ser promovido se superar o baseline em Brier e disagreement.

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

A investigação agora deve comparar:

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
threshold_recovery
second_mark_meta
double_value
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
T1T2 ou T1T3
   ↓
constraints
   ↓
ticket final
```

Essa separação permite descobrir se o ganho vem da escolha dos jogos duplos ou da escolha da segunda marcação.

---

# Error Recovery Score

O recovery usa apenas concursos anteriores e somente partidas em que o Top1 realmente falhou.

Definições:

```text
recovery_top2 = P(Top2_hit | Top1_miss, contexto)
recovery_top3 = P(Top3_hit | Top1_miss, contexto)
```

Como Top2 e Top3 são os únicos resultados possíveis quando Top1 erra:

```text
recovery_top2 + recovery_top3 ≈ 1
```

A primeira implementação usa contexto deliberadamente simples e regularizado.

---

# Estado atual do Second-Mark Disagreement

Resultado walk-forward atual:

```text
[SECOND-MARK DISAGREEMENT]
739 casos
Top2 baseline wins: 368
recovery wins:      371
recovery win rate:  50.20%
seletor final:      top2_baseline
```

Interpretação:

```text
Top2 baseline: 49.80%
Recovery:      50.20%
Diferença:     3 casos informativos
```

Esse resultado é **empate prático**, não evidência suficiente para promover `recovery` ao ticket final.

Ainda assim, ele é mais promissor que as correções históricas do Top1, pois foi a primeira linha histórica recente que não ficou claramente abaixo do baseline.

Critérios antes de promoção:

```text
IC95% do win rate
estabilidade temporal
ganho em P13+
ganho em P12+
robustez por faixa/contexto
```

Até lá:

```text
SecondMarkSelector final = top2_baseline
```

---

# Recovery Advantage

Criar explicitamente:

```text
recovery_advantage = recovery_top3 - recovery_top2
```

E também:

```text
probability_advantage = p(Top2) - p(Top3)
```

A troca `T2 → T3` não deve ocorrer por qualquer diferença mínima no histórico.

Regra candidata:

```text
usar T1T2 por padrão
usar T1T3 somente se recovery_advantage >= threshold
```

---

# Threshold Recovery

Testar thresholds em walk-forward:

```text
0.00
0.02
0.05
0.10
0.15
```

Saída desejada:

```text
threshold   trocas_T2_T3   Top2_wins   recovery_wins   win_rate   P13+   P12+
0.00              ...          ...            ...          ...      ...    ...
0.02              ...          ...            ...          ...      ...    ...
0.05              ...          ...            ...          ...      ...    ...
0.10              ...          ...            ...          ...      ...    ...
0.15              ...          ...            ...          ...      ...    ...
```

O objetivo é descobrir se o histórico só agrega valor quando sua preferência por Top3 é forte.

---

# Segmentação por gap Top2–Top3

Criar:

```text
gap_23 = p(Top2) - p(Top3)
```

Faixas iniciais:

```text
0–2 p.p.
2–5 p.p.
5–10 p.p.
10+ p.p.
```

Hipótese a testar:

> o histórico pode ter mais valor quando `p(Top2)` e `p(Top3)` estão muito próximos.

Exemplo de telemetria desejada:

```text
[SECOND-MARK BY GAP23]
0–2 p.p.   → recovery win rate ...
2–5 p.p.   → recovery win rate ...
5–10 p.p.  → recovery win rate ...
10+ p.p.   → recovery win rate ...
```

---

# Segmentação por p(Top1)

Também separar o Second-Mark Disagreement por confiança do Top1:

```text
33–40%
40–45%
45–50%
50–60%
60%+
```

O recovery pode ser neutro globalmente e útil apenas em partidas muito equilibradas.

Qualquer regra condicional deve ser definida apenas com histórico passado e testada prospectivamente.

---

# Melhorias do contexto de recovery

O contexto inicial é propositalmente simples.

Features candidatas para testes incrementais:

```text
p_top1
p_top2
p_top3
margin_top1_top2
gap_top2_top3
ratio_top3_top2
entropy
top1_is_1/X/2
```

Evitar adicionar tudo de uma vez para não criar buckets esparsos.

Ordem sugerida:

```text
1. gap Top2-Top3;
2. entropia;
3. p(Top2)/p(Top3);
4. identidade do Top1;
5. perfil probabilístico do concurso.
```

Toda expansão deve demonstrar ganho em walk-forward.

---

# Second-Mark Meta Model

Depois dos testes com recovery simples/threshold, testar um modelo específico para a segunda marcação.

Treinar somente nos casos em que Top1 errou.

Target:

```text
0 = Top2 foi o resultado real
1 = Top3 foi o resultado real
```

Features candidatas:

```text
p_top1
p_top2
p_top3
margin_top1_top2
gap_top2_top3
ratio_top3_top2
entropy
top1_is_X
top1_is_2
```

Saída:

```text
p_top3_given_top1_miss = P(Top3_hit | Top1_miss, contexto)
```

Regra básica:

```text
p_top3_given_top1_miss > threshold
→ T1T3

caso contrário
→ T1T2
```

O threshold deve ser escolhido em walk-forward, não necessariamente 0.50.

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
Top1+recovery
Top1+threshold_recovery
Top1+second_mark_meta
Top1+double_value
```

---

# Backtest matricial — Allocator × SecondMarkSelector

Comparar sistematicamente:

```text
                     Top2   Recovery   Threshold   Meta   DoubleValue
gain                  ...      ...         ...      ...       ...
uncertainty           ...      ...         ...      ...       ...
margin                ...      ...         ...      ...       ...
ratio                 ...      ...         ...      ...       ...
exact                 ...      ...         ...      ...       ...
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

Perguntas que esse backtest deve responder:

```text
qual allocator funciona melhor?
qual SecondMarkSelector funciona melhor?
qual combinação produz mais 13+?
o ganho vem do allocator ou da segunda marcação?
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

As constraints devem valer em treino, backtest e previsão final.

---

# Soft Constraint — Palmeiras

Favorecer, quando o custo probabilístico for pequeno, soluções que excluam a vitória do **PALMEIRAS/SP**.

Limiar atual:

```text
0.03
```

Esse valor é experimental e deve ser otimizado apenas com validação walk-forward.

---

# Estratégias atuais de alocação dos duplos

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

No baseline atual:

```text
Seco  = Top1
Duplo = Top1 + Top2
```

---

# Walk-forward validation

A seleção é feita sem informação futura:

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

`uncertainty` continua selecionada porque empata em 13+ com `gain`, `ratio` e `exact`, mas apresenta melhor P12+ entre elas.

---

# P13+ e P12+ empíricos

```text
P13+ empírico = concursos com 13 ou 14 / concursos testados
P12+ empírico = concursos com 12, 13 ou 14 / concursos testados
```

Resultados concurso a concurso:

```text
output/backtest.csv
```

Essas medidas devem permanecer separadas das probabilidades teóricas Poisson-binomial do ticket.

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

Para o recovery, testar especificamente:

```text
H0: recovery win rate = 50%
```

O atual `50.20%` não deve ser interpretado como vantagem sem intervalo de confiança.

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

O ECE baixo reforça que o baseline probabilístico é forte.

Calibração aplicada futura:

```text
exact_raw
vs
exact_calibrated
```

Só manter calibração aplicada se melhorar walk-forward.

---

# Distribution Backtest

Somente depois de entender melhor o valor da segunda marcação, ampliar o espaço das 19 marcações.

Distribuições candidatas:

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

---

# FullMarkingOptimizer

O FullMarkingOptimizer só deve entrar depois de validar onde existe sinal incremental.

Espaço futuro:

```text
Seco:  T1 | T2 | T3
Duplo: T1T2 | T1T3 | T2T3
```

Primeiro validar:

```text
T1T2 vs T1T3
```

Somente depois considerar:

```text
T2T3
Seco T2
Seco T3
```

---

# Telemetria desejada

Por jogo:

```text
p_top1
p_top2
p_top3
margin_top1_top2
gap_top2_top3
entropy
recovery_top2
recovery_top3
recovery_advantage
second_mark_selector
second_mark_choice
second_mark_threshold
second_mark_disagreement
```

Agregados:

```text
Top1 baseline accuracy
Top1 disagreement win rates
Second-Mark disagreement cases
Top2 baseline wins
Recovery wins
Recovery win rate
Recovery win rate por threshold
Recovery win rate por gap23
Recovery win rate por faixa de p(top1)
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

No PowerShell:

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
- [x] Top1 Disagreement Test;
- [x] `top1_residual`;
- [x] `top1_lift`;
- [x] `top1_reliability`;
- [x] `p(top1_meta)`;
- [x] evidência de que as correções testadas não superam o Top1;
- [x] `error_recovery_score` inicial;
- [x] `Second-Mark Disagreement Test`;
- [x] resultado inicial de recovery = `50.20%`;
- [x] manutenção de `top2_baseline` como seletor final.

## Próximas prioridades — ordem prática

1. [ ] implementar `recovery_advantage` explicitamente;
2. [ ] testar thresholds `T2 → T3` em walk-forward;
3. [ ] segmentar por `gap_23 = p(Top2)-p(Top3)`;
4. [ ] segmentar recovery por faixa de `p(Top1)`;
5. [ ] bootstrap + IC95% do Second-Mark Disagreement;
6. [ ] medir estabilidade temporal do recovery;
7. [ ] ampliar contexto de recovery incrementalmente;
8. [ ] implementar `second_mark_meta`;
9. [ ] implementar `double_value_score`;
10. [ ] criar backtest matricial `Allocator × SecondMarkSelector`;
11. [ ] medir impacto de cada seletor em P13+/P12+;
12. [ ] otimizar threshold final de troca T2→T3;
13. [ ] validar definitivamente `p(top1_meta)` com refit completo/features reduzidas;
14. [ ] implementar `distribution_backtest` Top1/Top2/Top3;
15. [ ] implementar FullMarkingOptimizer;
16. [ ] avaliar T2T3 e secos Top2/Top3 somente com evidência forte;
17. [ ] implementar KNN por similaridade;
18. [ ] implementar calibração aplicada;
19. [ ] testar `historical_13plus_score` como métrica complementar;
20. [ ] adicionar baseline aleatório;
21. [ ] remover/substituir desempate posicional arbitrário do `exact`;
22. [ ] otimizar o limiar do Palmeiras.

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
superar Top2 nos casos de discordância
↓
mostrar vantagem seletiva por threshold/contexto
↓
melhorar P13+ / P12+
↓
manter estabilidade temporal
↓
resistir a IC95% / bootstrap
```

A unidade final de avaliação continua sendo o ticket completo de 19 marcações.

---

# Princípio geral

```text
p(Top1) baseline
      +
DoubleAllocator
      +
SecondMarkSelector
      +
Recovery seletivo validado
      +
Validação walk-forward
      +
Incerteza estatística
      +
Constraints
      +
Otimização
      ↓
PALPITE FINAL
```

> **O histórico não precisa vencer o Top1 para ser útil: pode gerar valor escolhendo melhor qual resultado deve acompanhá-lo nos cinco duplos.**
