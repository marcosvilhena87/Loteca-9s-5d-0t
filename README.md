# Loteca ML — Estratégia 9 Secos / 5 Duplos / 0 Triplos

Projeto de Machine Learning para gerar **um único palpite final da Loteca**, com foco em maximizar a probabilidade de atingir **13 ou 14 acertos**, respeitando sempre:

- **9 secos**;
- **5 duplos**;
- **0 triplos**;
- **19 marcações** no total.

O projeto combina probabilidades por partida, ranking Top1/Top2/Top3, walk-forward, hard/soft constraints, backtesting, oráculos diagnósticos e otimização do ticket.

> O objetivo principal não é maximizar accuracy jogo a jogo. A unidade final é o **ticket completo de 19 marcações**, com prioridade para **P(>=13)**.

---

# Objetivo e dados

Arquivos principais:

```text
data/concursos_anteriores.csv
data/proximo_concurso.csv
```

Probabilidades normalizadas:

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

Base atual:

```text
445 concursos
30 concursos na janela histórica inicial
415 concursos testados em walk-forward

Top1: 51.7817%
Top2: 26.5329%
Top3: 21.6854%
```

---

# Função objetivo

A hierarquia de decisão é orientada à cauda superior:

```text
1. maior P13+
2. maior número de concursos com 14
3. maior número de concursos com 13
4. maior P12+
5. maior número de concursos com 12
6. maior média de acertos
7. menor instabilidade/variância
```

Accuracy, média, win rate, Brier Score, Log Loss e ECE são principalmente métricas diagnósticas. Uma alteração só deve ser promovida quando melhorar o **ticket fora da amostra**.

---

# Princípio central — preservar Top1 enquanto houver evidência para isso

O `p(Top1)` continua sendo o baseline individual mais forte.

Critérios mínimos para substituir/reordenar Top1:

```text
1. superar p(Top1) em walk-forward;
2. vencer quando houver discordância;
3. melhorar P13+ do ticket;
4. apresentar estabilidade temporal;
5. não usar informação futura.
```

Benchmarks atuais:

```text
[DISAGREEMENT] top1_residual
3234 casos | baseline 802 x histórico 750 | neutros 1682 | win rate 48.32%

[DISAGREEMENT] top1_lift
3264 casos | baseline 801 x histórico 755 | neutros 1708 | win rate 48.52%

[DISAGREEMENT] top1_reliability
3323 casos | baseline 821 x histórico 746 | neutros 1756 | win rate 47.61%

[TOP1-META]
Brier baseline: 0.233977
Brier meta:     0.240629

[DISAGREEMENT] p_top1_meta
4107 casos | baseline 1140 x meta 896 | neutros 2071 | win rate 44.01%
```

Conclusão:

> `top1_residual`, `top1_lift`, `top1_reliability` e `p(top1_meta)` permanecem como benchmarks/telemetria e não alteram o ticket final.

Essa conclusão vale para o **baseline seguro**. O projeto também passará a testar um espaço XYZ mais amplo, no qual Top1 não precisa necessariamente estar presente nos 14 jogos; qualquer promoção desse espaço exige nested walk-forward e evidência robusta.

---

# Arquitetura baseline

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

## DoubleAllocator

Políticas atuais:

```text
gain
top2_probability
uncertainty
margin
ratio
hist_top1
hist_top2
exact
```

Scores principais:

```text
gain / top2_probability = p(Top2)
uncertainty              = 1 - p(Top1)
margin                   = 1 - (p(Top1) - p(Top2))
ratio                    = p(Top2) / p(Top1)
```

`exact` avalia as `C(14,5) = 2.002` posições possíveis dos cinco duplos e maximiza principalmente `P(>=13)` segundo as probabilidades disponíveis.

Distinguir:

```text
exact_probability = otimização ex-ante usando probabilidades
oracle_allocator  = diagnóstico ex-post usando resultados reais
```

## SecondMarkSelector

Candidatos:

```text
top2_baseline
recovery
threshold_recovery
second_mark_meta
double_value
```

---

# Estado atual do backtest

415 concursos fora da janela inicial:

```text
gain / top2_probability
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

`uncertainty` permanece selecionada por desempate operacional. Ainda não existe separação robusta em P13+ entre `gain`, `top2_probability`, `uncertainty`, `ratio` e `exact`.

Telemetria:

```text
[ALLOCATOR OVERLAP]
uncertainty x gain:             4.299 / 5
uncertainty x top2_probability: 4.299 / 5
uncertainty x ratio:            4.728 / 5
uncertainty x exact:            4.658 / 5

[PAIRWISE] gain vs uncertainty
62 vitórias | 300 empates | 53 derrotas | delta médio +0.0217
```

---

# Error Recovery Score

```text
recovery_top2 = P(Top2_hit | Top1_miss, contexto)
recovery_top3 = P(Top3_hit | Top1_miss, contexto)
```

Resultado atual:

```text
[SECOND-MARK DISAGREEMENT]
739 casos | Top2 368 x recovery 371 | recovery win rate 50.20%
seletor final: top2_baseline
```

Thresholds:

```text
0.00 → 50.20% | IC95% 46.82%–53.86%
0.02 → 49.48% | IC95% 45.74%–53.36%
0.05 → 52.28% | IC95% 48.45%–56.47%
0.10 → 52.77% | IC95% 48.30%–57.23%
0.15 → 52.92% | IC95% 47.66%–58.19%
```

Nested recovery:

```text
Top2 baseline:   P13+ 1.4458% | P12+ 6.0241% | média 8.7205
Nested recovery: P13+ 1.2048% | P12+ 8.6747% | média 8.7759

delta P13+: -0.2410 p.p.
delta P12+: +2.6506 p.p.
```

Conclusão:

> O recovery atual melhora P12+ e média, mas reduz P13+. Portanto, `top2_baseline` permanece ativo.

---

# Oracle Decomposition — implementado

Os oráculos usam resultados reais **somente para diagnóstico retrospectivo** e nunca alimentam a previsão final.

```text
[ORACLE DECOMPOSITION]

baseline
P13+:  1.45% | P12+:  6.02% | média  8.7205

allocator oracle
P13+: 11.08% | P12+: 31.08% | média 10.7229

selector oracle
P13+:  5.54% | P12+: 21.45% | média 10.1831

full oracle
P13+: 41.93% | P12+: 65.06% | média 12.0289
```

Regret:

```text
[REGRET ALLOCATOR]
média 2.0024 | zero 8.92% | 2+ 67.23% | máximo 5

[REGRET SELECTOR]
média 1.4627 | zero 18.07% | 2+ 44.34% | máximo 4

[REGRET FULL]
média 3.3084 | zero 0.96% | 2+ 95.42% | máximo 5
```

---

# DistributionBacktest seguro — implementado

Nesta fase Top1 permanece coberto nos 14 jogos e as cinco marcações extras são distribuídas entre Top2 e Top3.

```text
14/5/0
14/4/1
14/3/2
14/2/3
14/1/4
14/0/5
```

Resultado atual:

```text
[DISTRIBUTION BACKTEST]

14/5/0: P13+ 1.45% | P12+ 5.54% | média 8.7446
14/4/1: P13+ 0.48% | P12+ 6.02% | média 8.7446
14/3/2: P13+ 0.72% | P12+ 6.27% | média 8.7639
14/2/3: P13+ 1.20% | P12+ 5.78% | média 8.7807
14/1/4: P13+ 1.20% | P12+ 5.30% | média 8.7590
14/0/5: P13+ 1.69% | P12+ 5.30% | média 8.6940
```

Leitura atual:

```text
melhor P13+ observado: 14/0/5
melhor P12+ observado: 14/3/2
melhor média:          14/2/3
```

A vantagem histórica de `14/0/5` sobre `14/5/0` em P13+ é pequena e **não pode ser promovida sem nested/robustez estatística**.

## OracleDistribution

```text
P13+: 41.69%
P12+: 64.34%
```

Muito próximo de:

```text
OracleFull P13+: 41.93%
```

Isso sugere que grande parte do teto pode ser expressa pela decisão conjunta de quantos Top2/Top3 usar e em quais jogos colocá-los.

Próximas telemetrias:

```text
[ORACLE DISTRIBUTION USAGE]
14/5/0: ...
14/4/1: ...
14/3/2: ...
14/2/3: ...
14/1/4: ...
14/0/5: ...

[DISTRIBUTION REGRET]
...
```

Quando uma Hard Constraint alterar uma composição nominal, registrar:

```text
requested_distribution
effective_distribution
constraint_adjusted
```

---

# NestedDistributionSelector

Não selecionar uma distribuição usando o mesmo período em que ela é avaliada.

```text
histórico até N
      ↓
comparar distribuições somente no passado
      ↓
selecionar
      ↓
congelar
      ↓
aplicar no concurso N+1
      ↓
registrar
      ↓
repetir
```

Telemetria:

```text
[NESTED DISTRIBUTION]
usage 14/5/0: ...
usage 14/4/1: ...
usage 14/3/2: ...
usage 14/2/3: ...
usage 14/1/4: ...
usage 14/0/5: ...

baseline P13+: ...
nested P13+:   ...
delta P13+:    ...
```

Somente o nested pode promover uma distribuição diferente do baseline seguro.

---

# Nova linha de pesquisa — XYZDistributionBacktest

Além do espaço seguro `14/x/y`, testar distribuições gerais das **19 marcações** entre Top1, Top2 e Top3.

Definição:

```text
X = quantidade total de marcações Top1
Y = quantidade total de marcações Top2
Z = quantidade total de marcações Top3

X + Y + Z = 19
```

Ponto central inicial:

```text
9/5/5
```

Esse ponto distribui as 19 marcações de forma mais equilibrada entre os três ranks e, ao contrário do espaço seguro, **não exige Top1 nos 14 jogos**.

Isso significa que o otimizador XYZ pode usar:

```text
SECO:
T1
T2
T3

DUPLO:
T1T2
T1T3
T2T3
```

Sempre respeitando:

```text
14 jogos
9 secos
5 duplos
0 triplos
19 marcações
```

---

# Vizinhança ±1 partindo de 9/5/5

A operação básica é transferir **uma marcação de uma coluna para outra**, preservando `X+Y+Z=19`.

Raio 0:

```text
9/5/5
```

Raio 1:

```text
9/5/5   baseline central
10/4/5  +1 Top1 / -1 Top2
10/5/4  +1 Top1 / -1 Top3
8/6/5   -1 Top1 / +1 Top2
8/5/6   -1 Top1 / +1 Top3
9/6/4   +1 Top2 / -1 Top3
9/4/6   -1 Top2 / +1 Top3
```

Gerador conceitual:

```text
(X+1,Y-1,Z)
(X+1,Y,Z-1)
(X-1,Y+1,Z)
(X,Y+1,Z-1)
(X-1,Y,Z+1)
(X,Y-1,Z+1)
```

Descartar automaticamente:

```text
X < 0
Y < 0
Z < 0
X + Y + Z != 19
distribuições incompatíveis com 9 secos / 5 duplos
```

Também remover duplicatas.

---

# Busca em raios XYZ

Depois do raio 1, expandir progressivamente:

```text
raio 0 → 9/5/5
raio 1 → vizinhos por uma transferência unitária
raio 2 → vizinhos dos vizinhos
raio 3 → expansão adicional somente se houver sinal
```

Não saltar diretamente para todo o espaço combinatório. A expansão gradual reduz custo e risco de selecionar retrospectivamente uma distribuição sortuda.

Cada distribuição deve receber um identificador canônico:

```text
XYZ_09_05_05
XYZ_10_04_05
XYZ_10_05_04
...
```

---

# XYZDistributionBacktest

Para cada distribuição XYZ viável, otimizar **também o posicionamento** das marcações nos 14 jogos.

Exemplo `9/5/5`:

```text
9 ocorrências de Top1
5 ocorrências de Top2
5 ocorrências de Top3
```

Não basta contar os ranks; é necessário escolher:

```text
quais jogos recebem Top1
quais jogos recebem Top2
quais jogos recebem Top3
quais 5 jogos recebem duas marcações
quais 9 jogos recebem uma marcação
```

Restrições estruturais:

```text
cada jogo recebe 1 ou 2 marcações
exatamente 5 jogos recebem 2 marcações
exatamente 9 jogos recebem 1 marcação
nenhum jogo recebe 0 ou 3 marcações
```

Telemetria inicial:

```text
[XYZ DISTRIBUTION BACKTEST]

X/Y/Z      14   13   12    P13+     P12+     mean
9/5/5       ...  ...  ...     ...       ...      ...
10/4/5      ...  ...  ...     ...       ...      ...
10/5/4      ...  ...  ...     ...       ...      ...
8/6/5       ...  ...  ...     ...       ...      ...
8/5/6       ...  ...  ...     ...       ...      ...
9/6/4       ...  ...  ...     ...       ...      ...
9/4/6       ...  ...  ...     ...       ...      ...
```

Registrar também:

```text
wins_vs_9_5_5
ties_vs_9_5_5
losses_vs_9_5_5
P13+ wins/ties/losses
P12+ wins/ties/losses
mean_delta_hits
```

---

# Viabilidade estrutural XYZ

Uma distribuição `X/Y/Z` só é válida se existir pelo menos uma atribuição aos 14 jogos que satisfaça simultaneamente:

```text
X + Y + Z = 19
9 secos
5 duplos
0 triplos
5 jogos com duas marcações
9 jogos com uma marcação
```

O validador deve verificar a viabilidade **antes** do backtest.

Funções sugeridas:

```text
is_xyz_distribution_valid(X, Y, Z)
generate_xyz_neighbors(X, Y, Z)
generate_xyz_radius(center=(9,5,5), radius=R)
```

---

# Otimizador de posicionamento XYZ

O `XYZDistributionBacktest` precisa resolver a atribuição das marcações aos jogos.

Espaço de marcação por jogo:

```text
T1
T2
T3
T1T2
T1T3
T2T3
```

Objetivo ex-ante inicial:

```text
maximizar score do ticket usando somente probabilidades pré-jogo
```

Estratégias de implementação possíveis:

```text
Programação Dinâmica
Branch and Bound
Integer Programming
busca exata com poda
```

Preferência inicial: **Programação Dinâmica**, porque o estado global pode rastrear contagens restantes de Top1/Top2/Top3, secos e duplos.

Status: **implementado** em `scripts/train_model.py`. O otimizador avalia as seis
marcações possíveis por jogo, maximiza a probabilidade total coberta sem acessar
resultados reais e incorpora a vitória do Flamengo durante a transição dos
estados. Assim, uma composição incompatível com a Hard Constraint é rejeitada
explicitamente, sem ajuste silencioso das contagens XYZ.

Também estão disponíveis o validador estrutural, o identificador canônico e os
geradores determinísticos de vizinhos e de raios. O espaço XYZ permanece
experimental e não altera o palpite final até obter evidência nested
walk-forward robusta.

Estado conceitual:

```text
(i, used_T1, used_T2, used_T3, used_doubles)
```

Ao final exigir:

```text
used_T1 = X
used_T2 = Y
used_T3 = Z
used_doubles = 5
```

---

# OracleXYZ

Criar um teto retrospectivo específico para o novo espaço.

Para cada concurso:

```text
qual distribuição XYZ permitida teria produzido mais acertos?
```

Registrar:

```text
[ORACLE XYZ]
P13+:
P12+:
mean:
```

E uso por distribuição:

```text
[ORACLE XYZ USAGE]
9/5/5: ...
10/4/5: ...
10/5/4: ...
8/6/5: ...
8/5/6: ...
9/6/4: ...
9/4/6: ...
...
```

O oracle é **diagnóstico apenas** e nunca pode alimentar diretamente a previsão final.

---

# NestedXYZDistributionSelector

Não promover a melhor distribuição XYZ observada retrospectivamente.

Fluxo:

```text
histórico até N
      ↓
gerar espaço XYZ permitido pelo raio atual
      ↓
comparar apenas no passado
      ↓
selecionar X/Y/Z
      ↓
congelar
      ↓
aplicar no concurso N+1
      ↓
registrar
      ↓
repetir
```

Telemetria:

```text
[NESTED XYZ]
center: 9/5/5
radius: 1

usage 9/5/5: ...
usage 10/4/5: ...
usage 10/5/4: ...
usage 8/6/5: ...
usage 8/5/6: ...
usage 9/6/4: ...
usage 9/4/6: ...

baseline P13+:
XYZ nested P13+:
delta P13+:

baseline P12+:
XYZ nested P12+:
delta P12+:
```

Somente após o raio 1 demonstrar ganho robusto fora da amostra considerar raio 2.

---

# Busca local / Hill Climbing XYZ — experimental

Uma alternativa ao teste completo do raio é busca local:

```text
best = 9/5/5
      ↓
testar vizinhos ±1
      ↓
se houver melhoria robusta, mover para o melhor vizinho
      ↓
repetir até não haver melhoria
```

Regra importante:

> a decisão de mover para um vizinho deve usar somente histórico anterior ao concurso testado.

Não usar hill climbing retrospectivo sobre os 415 concursos completos para definir a distribuição final.

---

# Comparação segura × XYZ

O espaço XYZ é mais agressivo que o espaço `14/x/y`, pois pode remover Top1 e introduzir `T2T3`.

Comparações obrigatórias:

```text
uncertainty + top2_baseline
melhor distribuição segura nested
joint_probability
XYZ 9/5/5
NestedXYZ raio 1
```

Registrar:

```text
P13+
P12+
mean
stddev
regret
pairwise P13+
pairwise P12+
bootstrap IC95%
```

A distribuição XYZ só pode ser promovida se superar o baseline seguro **fora da amostra**.

---

# Top1-only baseline e valor das cinco marcas extras

Adicionar explicitamente:

```text
[TOP1 ONLY]
P13+:
P12+:
mean:
```

## Extra Mark Efficiency

```text
extra_mark_efficiency = (hits_ticket - hits_top1_only) / 5
```

## Oracle Capture Rate

```text
capture_rate =
    (hits_policy - hits_top1_only)
    /
    (hits_oracle_full - hits_top1_only)
```

Quando não houver ganho oracle disponível, registrar `no_oracle_gain_available`.

---

# Recovery Profile por concurso

Adicionar:

```text
top1_hits
top1_misses
recoverable_by_top2
recoverable_by_top3
recoverable_by_either
```

Também segmentar:

```text
Top1 fez 10+
Top1 fez 9
Top1 fez 8
Top1 fez 7
Top1 fez <=6
```

---

# Opportunity Dataset

Criar:

```text
output/opportunity_dataset.csv
```

Campos principais:

```text
contest
game
p_top1
p_top2
p_top3
gap_12
gap_23
entropy
ratio_top2_top1
ratio_top3_top1
top1_result
top2_result
top3_result
top1_hit
recoverable_by_top2
recoverable_by_top3
```

Targets locais:

```text
top1_miss
extra_gain_top2
extra_gain_top3
```

Separação desejada:

```text
modelo local de valor
        +
otimização global das vagas/marcações
```

---

# JointMarkAllocator

Arquitetura conjunta paralela à sequência `DoubleAllocator → SecondMarkSelector`.

Para cada jogo:

```text
Jogo i → T1T2 → score_T2
Jogo i → T1T3 → score_T3
```

Selecionar exatamente 5 oportunidades, respeitando no máximo uma por jogo.

## Baseline joint_probability

Antes de novo ML:

```text
score_T2 = p(Top2)
score_T3 = p(Top3)
```

Selecionar globalmente as cinco melhores oportunidades entre as 28 possibilidades, com no máximo uma por jogo.

Comparar:

```text
uncertainty + top2_baseline
14/0/5 fixo
NestedDistributionSelector
joint_probability
```

---

# DoubleValueModel

Depois de `joint_probability`, aprender scores locais.

Features candidatas:

```text
p_top1
p_top2
p_top3
gap_12
gap_23
ratio_top2_top1
ratio_top3_top1
entropy
identidade Top1/Top2/Top3
posição do jogo
perfil probabilístico do concurso
```

Saídas:

```text
score_T2 ≈ P(extra_gain_top2 = 1 | contexto)
score_T3 ≈ P(extra_gain_top3 = 1 | contexto)
```

Todo treinamento e hiperparâmetro deve ser escolhido em walk-forward/nested walk-forward.

---

# Hard Constraints

Todo ticket deve conter exatamente:

```text
14 jogos
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

## Palmeiras — Soft Constraint

Favorecer, quando o custo probabilístico for pequeno, soluções que excluam a vitória do **PALMEIRAS/SP**.

Limiar atual:

```text
0.03
```

No espaço XYZ, as constraints devem ser incorporadas ao otimizador sem violar as contagens X/Y/Z. Se a Hard Constraint tornar a distribuição nominal inviável, registrar explicitamente ajuste ou inviabilidade; nunca alterar silenciosamente a composição.

---

# Testes automatizados obrigatórios

Garantir permanentemente:

```text
14 jogos por concurso
9 secos
5 duplos
0 triplos
19 marcações
vitória do Flamengo coberta
probabilidades somando 1
Top1/Top2/Top3 distintos
desempate 1 > 2 > X
nenhum vazamento temporal
```

No espaço seguro:

```text
Top1 coberto nos 14 jogos
```

Para XYZ:

```text
X + Y + Z = 19
contagem real de T1 = X
contagem real de T2 = Y
contagem real de T3 = Z
9 secos
5 duplos
nenhum triplo
nenhum jogo sem marcação
nenhuma marca repetida no mesmo jogo
```

Para vizinhança:

```text
cada vizinho difere por exatamente uma transferência unitária entre duas colunas
soma permanece 19
duplicatas removidas
```

Em toda rotina histórica:

```python
assert train_contest < test_contest
```

Oráculos:

```text
nunca alimentar previsão final
nunca ser usados diretamente como features pré-jogo
servir para diagnóstico e teto estrutural
```

---

# Controle de experimentos

Criar/manter:

```text
output/experiments.csv
```

Campos sugeridos:

```text
timestamp
model
search_space
xyz_center
xyz_radius
distribution
requested_distribution
effective_distribution
constraint_adjusted
allocator
second_mark_selector
optimizer
window
decay
features
n14
n13
n12
P13+
P12+
mean
stddev
oracle_capture_rate
git_commit
```

---

# Telemetria resumida

```text
[SUMMARY]
Top1 accuracy:
Top1-only mean/P13+:
Selected safe distribution:
Selected XYZ distribution:
XYZ radius:
Selected allocator/optimizer:
Historical P13+:
Best experimental P13+:

Oracle allocator P13+:    11.08%
Oracle selector P13+:       5.54%
Oracle distribution P13+:  41.69%
Oracle full P13+:          41.93%
Oracle XYZ P13+:           ...

Nested safe P13+: ...
Nested XYZ P13+:  ...
Joint probability P13+: ...

Current contest P14:
Current contest P13:
Current contest P13+:
Current contest E[hits]:
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
│   ├── backtest.csv
│   ├── experiments.csv          # planejado
│   └── opportunity_dataset.csv  # planejado
├── tests/
└── README.md
```

---

# Execução

```powershell
python main.py
```

Testes:

```powershell
python -m unittest discover -v
```

---

# Roadmap — ordem prática

## Concluído

- [x] pipeline único de constraints;
- [x] invariantes 9/5/0 e 19 marcações;
- [x] políticas atuais de allocator;
- [x] métricas `P(14)`, `P(13)`, `P(>=13)` e `E[acertos]`;
- [x] walk-forward sem vazamento temporal;
- [x] benchmarks de Top1;
- [x] Error Recovery + thresholds + nested;
- [x] overlap e pairwise inicial;
- [x] `OracleAllocator`;
- [x] `OracleSecondMark`;
- [x] `OracleFull`;
- [x] regret allocator/selector/full;
- [x] `DistributionBacktest` seguro `14/5/0 → 14/0/5`;
- [x] otimização de posicionamento dentro de cada distribuição segura;
- [x] `OracleDistribution`.

## Fase 1 — validar distribuição segura

1. [ ] implementar `NestedDistributionSelector`;
2. [ ] registrar uso nested de cada distribuição;
3. [ ] registrar `OracleDistribution usage`;
4. [ ] calcular regret por distribuição;
5. [ ] pairwise P13+/P12+;
6. [ ] bootstrap pareado.

## Fase 2 — XYZ em torno de 9/5/5

7. [ ] implementar `is_xyz_distribution_valid()`;
8. [ ] implementar `generate_xyz_neighbors()`;
9. [ ] implementar `generate_xyz_radius()`;
10. [ ] implementar otimizador de posicionamento XYZ;
11. [ ] testar `9/5/5`;
12. [ ] testar raio 1: `10/4/5`, `10/5/4`, `8/6/5`, `8/5/6`, `9/6/4`, `9/4/6`;
13. [ ] implementar `XYZDistributionBacktest`;
14. [ ] implementar `OracleXYZ`;
15. [ ] registrar `OracleXYZ usage`;
16. [ ] implementar `NestedXYZDistributionSelector` raio 1;
17. [ ] pairwise/bootstrap XYZ vs baseline seguro;
18. [ ] só então decidir se raio 2 será testado.

## Fase 3 — valor das marcações extras

19. [ ] implementar `Top1-only baseline`;
20. [ ] implementar `Extra Mark Efficiency`;
21. [ ] implementar `Oracle Capture Rate`;
22. [ ] implementar `Recovery Profile`;
23. [ ] segmentar por quantidade de acertos/erros Top1.

## Fase 4 — otimização conjunta sem ML

24. [ ] implementar `JointMarkAllocator`;
25. [ ] implementar `joint_probability`;
26. [ ] comparar joint vs safe nested vs XYZ nested;
27. [ ] medir regret e Oracle Capture.

## Fase 5 — aprender valor marginal

28. [ ] criar `output/opportunity_dataset.csv`;
29. [ ] criar targets `extra_gain_top2` e `extra_gain_top3`;
30. [ ] implementar `DoubleValueModel`;
31. [ ] criar `joint_learned`;
32. [ ] validar em nested walk-forward.

## Fase 6 — robustez

33. [ ] expanding × rolling windows;
34. [ ] decay temporal em nested;
35. [ ] reliability tables Top1/Top2/Top3;
36. [ ] `output/experiments.csv`;
37. [ ] IC95%/bootstrap final das estratégias candidatas.

## Fase 7 — expansão global

38. [ ] expandir XYZ além do raio 1 somente se houver evidência;
39. [ ] avaliar busca local/hill climbing nested;
40. [ ] comparar espaço XYZ com FullMarkingOptimizer irrestrito;
41. [ ] remover/substituir desempates arbitrários do `exact`;
42. [ ] otimizar limiar do Palmeiras com validação adequada.

---

# Critério de promoção

Uma estratégia experimental só pode substituir o baseline quando:

```text
melhorar P13+ fora da amostra
↓
não depender de seleção retrospectiva de hiperparâmetros
↓
apresentar resultado pareado favorável
↓
apresentar IC/bootstrap compatível com ganho real
↓
manter estabilidade temporal
↓
respeitar todas as Hard Constraints
```

No espaço XYZ, acrescentar:

```text
superar o baseline seguro em nested
↓
justificar a remoção de Top1 em alguns jogos
↓
mostrar que o ganho não depende apenas de uma distribuição extrema sorteada no histórico
```

---

# Princípio geral

```text
Baseline seguro Top1
      +
Oracle Decomposition
      +
DistributionBacktest seguro
      +
NestedDistributionSelector
      +
XYZ 9/5/5 ± raio controlado
      +
NestedXYZDistributionSelector
      +
JointMarkAllocator / DoubleValueModel
      +
Regret / Pairwise / Bootstrap
      +
Hard Constraints
      +
Otimização de P13+
      ↓
PALPITE FINAL
```

> **Não promover a melhor regra observada; promover apenas a regra que continuar melhor quando escolhida usando somente o passado e avaliada prospectivamente.**
