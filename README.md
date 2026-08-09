# Loteca ML — Estratégia 9 Secos / 5 Duplos / 0 Triplos

Projeto de Machine Learning para gerar **um único palpite final da Loteca**, com foco em maximizar a chance de atingir **13 ou 14 acertos**, respeitando sempre:

- **9 secos**;
- **5 duplos**;
- **0 triplos**;
- **19 marcações** no total.

O projeto combina probabilidades por partida, ranking Top1/Top2/Top3, validação walk-forward, constraints, backtesting e otimização do ticket.

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

Na base atual:

```text
445 concursos
30 concursos na janela histórica inicial
415 concursos testados em walk-forward

Top1: 51.7817%
Top2: 26.5329%
Top3: 21.6854%
```

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

---

# Correções do Top1 — benchmarks congelados

Resultados atuais:

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

O `p(top1_meta)` ainda pode receber uma validação técnica final com features menos redundantes e refit completo do histórico em cada passo walk-forward, mas essa linha deixou de ser prioridade.

---

# Nova prioridade — recuperar erros do Top1

Como o Top1 permanece difícil de superar, a principal linha de pesquisa passa a ser:

> **quando o Top1 estiver errado, qual resultado deve acompanhá-lo no duplo para recuperar melhor esse erro?**

Baseline atual:

```text
Seco  = Top1
Duplo = Top1 + Top2
```

Nova comparação:

```text
T1T2
vs
T1T3
```

O Top1 continua coberto; apenas a segunda marcação é otimizada.

---

# Arquitetura — DoubleAllocator × SecondMarkSelector

O problema passa a ser dividido em duas decisões independentes.

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

# Error Recovery Score

O recovery é estimado usando somente concursos anteriores e somente jogos nos quais Top1 realmente falhou.

```text
recovery_top2 = P(Top2_hit | Top1_miss, contexto)
recovery_top3 = P(Top3_hit | Top1_miss, contexto)
```

Como Top2 e Top3 são os dois únicos resultados restantes quando Top1 erra:

```text
recovery_top2 + recovery_top3 ≈ 1
```

A primeira implementação usa contexto simples, suavizado e sem informação futura.

---

# Estado atual do Second-Mark Disagreement

Resultado global:

```text
[SECOND-MARK DISAGREEMENT]
739 casos
Top2 baseline wins: 368
recovery wins:      371
recovery win rate:  50.20%
seletor final:      top2_baseline
```

Isso representa apenas **+3 decisões líquidas** para recovery e deve ser tratado como empate prático.

---

# Threshold Recovery — resultados atuais

A troca `T2 → T3` passa a exigir vantagem mínima do recovery:

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

Definição:

```text
net_recovery_gain = recovery_wins - top2_wins
```

Leitura atual:

- thresholds `0.05`, `0.10` e `0.15` apresentam ganho líquido aparente;
- `0.10` produz o maior ganho líquido atual: **+26**;
- `0.15` produz o maior win rate atual: **52.92%**;
- **todos os IC95% ainda incluem 50%**;
- portanto, nenhum threshold está promovido ao ticket final.

A regra final permanece:

```text
SecondMarkSelector = top2_baseline
```

---

# Risco de overfitting do threshold

Não selecionar `0.10` ou `0.15` simplesmente porque foram melhores nos mesmos 415 concursos usados para avaliá-los.

Isso configuraria seleção do hiperparâmetro usando o próprio período de teste.

A próxima implementação prioritária deve usar **nested walk-forward**.

---

# Nested Walk-Forward para threshold

Fluxo desejado:

```text
histórico disponível até N
        ↓
comparar thresholds somente nesse passado
        ↓
escolher threshold vencedor
        ↓
aplicar somente no concurso N+1
        ↓
registrar resultado
        ↓
incluir N+1 no histórico
        ↓
repetir
```

Exemplo:

```text
Concursos 1..100 → escolher threshold
Concurso 101     → teste real
Concursos 1..101 → recalcular threshold
Concurso 102     → teste real
...
```

O resultado nested é o que deve decidir se `threshold_recovery` merece promoção.

## Resultado nested walk-forward

A seleção prospectiva já foi implementada. Cada score histórico usado para escolher
o threshold também é fora da amostra, e o threshold fica congelado antes de avaliar
o concurso seguinte.

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

Thresholds escolhidos usando somente o passado:

```text
0.05: 373 concursos
0.10:  42 concursos
0.15:   0 concursos
```

Embora o nested tenha melhorado P12+ e a média, reduziu P13+. Portanto, em respeito
ao critério de sucesso e às Hard Constraints metodológicas, ele **não foi promovido**:
o ticket final continua usando `top2_baseline`.

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

Hipótese principal:

> recovery pode ter mais valor quando Top2 e Top3 são probabilisticamente próximos.

Telemetria desejada:

```text
[SECOND-MARK BY GAP23]
faixa        trocas   Top2_wins   recovery_wins   win_rate   IC95%
0–2 p.p.       ...       ...           ...          ...       ...
2–5 p.p.       ...       ...           ...          ...       ...
5–10 p.p.      ...       ...           ...          ...       ...
10+ p.p.       ...       ...           ...          ...       ...
```

---

# Regra bidimensional Recovery × Gap23

Depois da análise univariada, testar regras como:

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

Total inicial:

```text
3 × 3 = 9 regras
```

A seleção dessas regras também deve ocorrer dentro do nested walk-forward.

---

# Segmentação por p(Top1)

Separar também por confiança do Top1:

```text
33–40%
40–45%
45–50%
50–60%
60%+
```

O objetivo é descobrir se recovery só agrega valor em jogos equilibrados ou em algum regime específico.

---

# Backtest ticket-level — teste decisivo

Win rate de segunda marcação não é suficiente.

A promoção precisa melhorar o **ticket completo**.

Comparação prioritária:

```text
uncertainty + Top2 baseline
vs
uncertainty + recovery threshold 0.05
vs
uncertainty + recovery threshold 0.10
vs
uncertainty + recovery threshold 0.15
vs
uncertainty + threshold escolhido em nested walk-forward
```

Registrar:

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

A pergunta final não é apenas:

> recovery escolheu melhor T2/T3?

mas:

> **essa escolha produziu mais tickets com 13 ou 14 acertos?**

---

# Backtest matricial — Allocator × SecondMarkSelector

Depois do ticket-level inicial, comparar:

```text
                 Top2   Rec .05   Rec .10   Rec .15   Nested

gain              ...      ...       ...       ...      ...
uncertainty       ...      ...       ...       ...      ...
margin            ...      ...       ...       ...      ...
ratio             ...      ...       ...       ...      ...
exact             ...      ...       ...       ...      ...
```

Cada célula deve registrar:

```text
14 / 13 / 12 / 11 / 10 / <=9
P13+
P12+
mean
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

Usar comparação pareada por concurso.

Implementar:

```text
bootstrap >= 1.000 reamostragens
IC95% de P13+
IC95% de P12+
IC95% de delta P13+
IC95% de net_recovery_gain
IC95% do Second-Mark win rate
```

Quando o intervalo incluir o baseline, registrar:

```text
estatisticamente indistinguível
```

---

# Estabilidade temporal

Medir por:

```text
primeiro terço
segundo terço
último terço
```

ou janelas móveis.

Registrar:

```text
threshold escolhido por período
Second-Mark win rate
net_recovery_gain
P13+
P12+
pior janela
melhor janela
```

Uma regra não deve ser promovida se todo o ganho estiver concentrado em um período curto.

---

# Melhorias do contexto de recovery

Somente depois de validar `gap_23` e thresholds, ampliar o contexto.

Features candidatas:

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

Ordem sugerida de experimentação:

```text
1. gap_23;
2. entropia;
3. ratio_top3_top2;
4. identidade do Top1;
5. perfil probabilístico do concurso.
```

Evitar adicionar várias dimensões simultaneamente para reduzir risco de buckets esparsos e overfitting.

---

# Second-Mark Meta Model

Somente depois dos testes de recovery simples/threshold.

Treinar apenas em jogos onde Top1 errou.

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
P(Top3_hit | Top1_miss, contexto)
```

Qualquer threshold desse modelo também deve ser escolhido por nested walk-forward.

---

# Double Value Score

Combinar probabilidade atual e recovery histórico:

```text
score_T2 = α × p(Top2) + (1-α) × recovery_top2
score_T3 = α × p(Top3) + (1-α) × recovery_top3
```

Grid inicial:

```text
α ∈ {0.00, 0.25, 0.50, 0.75, 1.00}
```

Também selecionar `α` somente dentro do histórico disponível em cada passo nested.

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

## hist_top1 / hist_top2

Benchmarks posicionais históricos.

## exact

Avalia:

```text
C(14,5) = 2.002
```

alocações dos cinco duplos e maximiza principalmente `P(>=13)`.

No baseline atual, o `exact` ainda assume:

```text
Seco  = Top1
Duplo = Top1 + Top2
```

---

# Estado atual do backtest

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

---

# Calibração

Diagnóstico atual:

```text
Brier multiclasse: 0.588408
Log Loss:          0.985557
ECE:               0.012378
```

A calibração permanece diagnóstica até demonstrar ganho no ticket em walk-forward.

---

# Distribution Backtest e FullMarkingOptimizer

Essas etapas ficam **depois** da validação da segunda marcação.

Espaço futuro:

```text
Seco:  T1 | T2 | T3
Duplo: T1T2 | T1T3 | T2T3
```

Antes de abrir completamente esse espaço, o projeto deve demonstrar que existe sinal robusto na troca seletiva `T1T2 → T1T3`.

---

# Telemetria desejada

Por jogo:

```text
p_top1
p_top2
p_top3
margin_top1_top2
gap_top2_top3
recovery_top2
recovery_top3
recovery_advantage
recovery_threshold
second_mark_baseline
second_mark_selected
second_mark_switched
```

Agregados:

```text
threshold
switches
Top2 wins
recovery wins
net_recovery_gain
Second-Mark win rate
IC95%
P13+
P12+
estabilidade temporal
```

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

# Roadmap — ordem prática

## Concluído

- [x] pipeline único de constraints;
- [x] invariantes 9/5/0;
- [x] políticas `gain`, `uncertainty`, `margin`, `ratio`, `hist_top1`, `hist_top2` e `exact`;
- [x] `P(14)`, `P(13)`, `P(>=13)` e `E[acertos]`;
- [x] walk-forward sem vazamento temporal;
- [x] backtest 10–14;
- [x] P13+ e P12+ empíricos;
- [x] `output/backtest.csv`;
- [x] Disagreement Test do Top1;
- [x] `top1_residual`, `top1_lift`, `top1_reliability`;
- [x] `p(top1_meta)`;
- [x] evidência para congelar as correções do Top1;
- [x] `error_recovery_score`;
- [x] Second-Mark Disagreement;
- [x] thresholds `0.00`, `0.02`, `0.05`, `0.10`, `0.15`;
- [x] IC95% por threshold;
- [x] identificação do ganho líquido aparente em `0.05–0.15`.

## Próximas prioridades

1. [x] implementar **nested walk-forward** para seleção do threshold;
2. [x] registrar `net_recovery_gain` formalmente na telemetria;
3. [x] segmentar por `gap_23`;
4. [ ] testar regra bidimensional `recovery_advantage × gap_23`;
5. [ ] segmentar por faixa de `p(Top1)`;
6. [ ] implementar backtest ticket-level dos thresholds;
7. [ ] implementar matriz `Allocator × SecondMarkSelector`;
8. [ ] bootstrap pareado por concurso;
9. [ ] medir estabilidade temporal do threshold;
10. [ ] melhorar `recovery_context` incrementalmente;
11. [ ] implementar `second_mark_meta`;
12. [ ] implementar `double_value_score`;
13. [ ] comparar todas as variantes em P13+/P12+;
14. [ ] implementar `distribution_backtest`;
15. [ ] implementar FullMarkingOptimizer somente após sinal robusto;
16. [ ] validar T2T3 e secos Top2/Top3 apenas se houver evidência;
17. [ ] remover/substituir desempate arbitrário do `exact`;
18. [ ] otimizar o limiar do Palmeiras.

---

# Critério de sucesso

Para Top1:

```text
preservar p(Top1) enquanto nenhum candidato demonstrar superioridade robusta
```

Para a segunda marcação:

```text
superar Top2 em nested walk-forward
↓
IC95% / bootstrap favorável
↓
net_recovery_gain positivo e estável
↓
melhorar P13+ / P12+
↓
manter estabilidade temporal
```

A unidade final continua sendo o **ticket completo de 19 marcações**.

---

# Princípio geral

```text
p(Top1) preservado
      +
DoubleAllocator
      +
SecondMarkSelector validado
      +
Nested Walk-Forward
      +
Incerteza estatística
      +
Hard Constraints
      +
Otimização
      ↓
PALPITE FINAL
```

> **Não promover a melhor regra observada; promover apenas a regra que continuar melhor quando escolhida usando somente o passado e avaliada prospectivamente.**
