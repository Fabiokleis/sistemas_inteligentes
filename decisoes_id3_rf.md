# Decisões de Projeto — ID3 + Random Forest

## 1. Problema

Classificar vítimas de acidentes em 4 classes de gravidade (1=crítico, 2=instável,
3=potencialmente estável, 4=estável) e estimar numericamente a gravidade.

Dados de entrada disponíveis por vítima:
- pSist: pressão sistólica [5, 22]
- pDiast: pressão diastólica [0, 15]
- qPA: qualidade da pressão arterial [-10, 10]
- pulso: batimentos por minuto [0, 200]
- resp: frequência respiratória [0, 22]
- gravidade: calculada a partir dos sinais vitais (fórmula perdida)
- classe: 1–4

---

## 2. Features utilizadas

**Usadas:** qPA, pulso, resp (índices 3, 4, 5 no arquivo)

**Descartadas:** pSist e pDiast

**Motivo:** o enunciado instrui explicitamente a não usar pSist e pDiast diretamente,
pois elas são usadas apenas no cálculo intermediário de qPA. Usar as três ao mesmo
tempo introduziria redundância e vazamento de informação.

---

## 3. Discretização

O ID3 é um algoritmo que trabalha com atributos categóricos: a cada nó da árvore,
ele pergunta "qual é o valor de X?" e cada resposta leva a um galho diferente. Mas
qPA, pulso e resp são números reais contínuos — há infinitos valores possíveis, o
que tornaria a árvore infinita. É preciso converter esses valores em categorias
discretas (faixas).

### Método escolhido: equal-width (largura igual)

O range de cada feature é dividido em N intervalos de tamanho igual. Um valor é
mapeado para o índice da faixa em que cai (0 a N-1). Valores fora do range são
fixados no extremo mais próximo (clamp).

Com N_FAIXAS = 7:

| Feature | Range     | Largura da faixa | Exemplo                              |
|---------|-----------|------------------|--------------------------------------|
| qPA     | [-10, 10] | 20 / 7 ≈ 2.86    | qPA = -8.5 → faixa 0 (mais crítica)  |
| pulso   | [0, 200]  | 200 / 7 ≈ 28.57  | pulso = 56 → faixa 1                 |
| resp    | [0, 22]   | 22 / 7 ≈ 3.14    | resp = 9.2 → faixa 2                 |

### Por que equal-width e não equal-frequency?

A alternativa mais comum é o **equal-frequency** (ou quantil): dividir os dados em
N grupos com o mesmo número de exemplos cada. A vantagem do equal-frequency é que
cada faixa tem representação balanceada no treino.

Para este problema, o **equal-width é a escolha mais adequada por três razões**:

1. **Preserva o significado médico dos intervalos.** Com equal-width, a faixa 0
   de qPA sempre corresponde a "pressão muito baixa" (entre -10 e -7.14). Com
   equal-frequency, a mesma faixa dependeria de quais valores caíram no treino —
   perdendo o significado físico do atributo.

2. **Independência do conjunto de treino.** O equal-frequency recalcula os limites
   das faixas com base na distribuição dos dados de treino. Se os dados de teste
   ou os dados reais de acidente tiverem uma distribuição ligeiramente diferente, os
   limites estariam "errados". Com equal-width e os ranges fornecidos pelo enunciado,
   qualquer valor futuro dentro do domínio médico é mapeado da mesma forma.

3. **O enunciado fornece os ranges exatos do domínio.** Como os dados são gerados
   de forma randômica dentro de intervalos conhecidos, os limites do equal-width são
   precisamente os limites físicos do problema — não uma aproximação.

---

## 4. Escolha de N_FAIXAS = 7

### O mecanismo do underfitting (N pequeno)

Com N=3, cada feature tem apenas 3 faixas. Para o qPA (range de 20 unidades), cada
faixa tem ~6.67 unidades de largura. Isso significa que uma vítima com qPA=-9
(pressão criticamente baixa) cai na **mesma faixa** que uma com qPA=-4 (pressão
apenas levemente abaixo do ideal). Para a árvore, esses dois casos são
indistinguíveis — ela precisa decidir a mesma classe para situações clinicamente
muito diferentes.

O resultado é que a árvore não consegue separar as classes com precisão suficiente,
pois as faixas largas misturam exemplos de classes distintas. Isso é **underfitting**:
o modelo é simples demais para capturar o padrão real nos dados.

### O mecanismo do overfitting (N grande)

Com N=20, cada faixa de qPA tem apenas 1 unidade de largura. A árvore aprende regras
ultra-específicas como "qPA entre -3.5 e -2.5 → classe 2". Essas regras descrevem
com precisão os exemplos do treino, mas são frágeis: um exemplo de teste com qPA=-3.6
cai na faixa vizinha e pode receber uma decisão completamente diferente, mesmo sendo
clinicamente equivalente.

Com muitas faixas, cada nó interno da árvore tem muitos filhos, e vários desses filhos
só contêm 1 ou 2 exemplos de treino. A árvore "decora" esses exemplos em vez de
aprender o padrão geral. Isso é **overfitting**: altíssima acurácia no treino (98%),
mas ruim no teste (66%) porque as regras não generalizam.

### Tabela e critério de escolha

Testamos N_FAIXAS ∈ {3, 5, 7, 10, 15, 20} com hold-out 80/20:

| N_FAIXAS | Acc Treino | Acc Teste | Interpretação          |
|----------|------------|-----------|------------------------|
| 3        | 68.33%     | 68.67%    | underfitting           |
| 5        | 83.33%     | 81.67%    | ainda generalizando    |
| **7**    | **86.67%** | **82.33%**| **melhor teste**       |
| 10       | 92.58%     | 80.67%    | início do overfitting  |
| 15       | 97.75%     | 75.67%    | overfitting moderado   |
| 20       | 98.00%     | 66.33%    | overfitting severo     |

O critério de escolha é sempre a **acurácia no conjunto de teste**, pois o treino
não é uma estimativa confiável de desempenho real. N=7 foi o ponto de equilíbrio
entre granularidade suficiente para separar as classes e generalização para dados
não vistos.

---

## 5. Algoritmo ID3

O ID3 constrói uma árvore de decisão escolhendo recursivamente o atributo que
maximiza o **ganho de informação** (IG).

**Entropia:** mede a "mistura" de classes num conjunto S
```
H(S) = - Σ p(c) * log2(p(c))
```
Máxima quando todas as classes são equiprováveis; zero quando o conjunto é puro.

**Ganho de informação:** redução de entropia ao dividir pelo atributo A
```
IG(S, A) = H(S) - Σ_v [ |S_v|/|S| * H(S_v) ]
```

**Casos base da recursão:**
1. Conjunto puro (todas as classes iguais) → folha com essa classe
2. Sem atributos disponíveis → folha com a classe majoritária

**Profundidade máxima:** 3 (uma divisão por feature; features não se repetem no ID3)

**Gravidade nas folhas:** a tarefa exige dois tipos de saída simultâneos:
classificação (classe 1–4) e regressão (gravidade, número real). O ID3 produz
naturalmente apenas classes. Para obter também a gravidade sem mudar a estrutura
do algoritmo, cada folha armazena a **média da gravidade** dos exemplos de treino
que chegaram a ela.

A lógica é: exemplos que chegam à mesma folha são, por definição, similares
(percorreram o mesmo caminho na árvore, portanto têm os mesmos valores de faixa
para qPA, pulso e resp). Logo, a média de gravidade desses exemplos é uma estimativa
razoável para qualquer novo exemplo que chegue à mesma folha. A predição sempre
retorna o par `(classe, gravidade_media)`.

Nos nós internos também guardamos a média de gravidade de todos os exemplos que
passaram por aquele nó. Isso serve como fallback: se um exemplo de teste tem um
valor de faixa que não apareceu no treino (um galho que não foi criado), usamos
a média de gravidade do nó pai como estimativa.

**Fallback para valores não vistos:** se um valor de faixa aparece no teste mas
não estava no treino, retorna a classe majoritária e a média de gravidade de todas
as folhas da sub-árvore.

---

## 6. Validação — Hold-out 80/20

**Método:** os 1500 exemplos são embaralhados (seed=42 para reprodutibilidade)
e divididos em 80% treino (1200) e 20% teste (300).

**Por que embaralhar?** Os dados podem estar ordenados de alguma forma. Sem
embaralhamento, o conjunto de teste poderia ter um perfil diferente do treino.

**Por que 20% de teste?** Equilíbrio clássico entre ter dados suficientes para
treinar e ter uma amostra confiável para avaliar.

**Por que não testar no arquivo sem label (01)?** Esse arquivo não tem a classe
real, portanto não é possível calcular acurácia nele.

**Métricas:**
- **Acurácia:** proporção de classes corretamente preditas (classificação)
- **RMSE:** raiz do erro quadrático médio na gravidade (regressão)

---

## 7. Random Forest

### O problema do ID3 sozinho: alta variância

Uma árvore de decisão é um modelo de **alta variância**: ela se ajusta muito ao
conjunto de treino específico que recebeu. Se o treino tivesse alguns exemplos
diferentes (o que aconteceria com outro split 80/20), a árvore poderia ter uma
estrutura completamente diferente. Isso é problemático porque não sabemos se a
árvore que treinamos reflete o padrão real dos dados ou apenas os exemplos que
vimos.

### A ideia do ensemble: "sabiamente juntos"

A solução é construir **muitas árvores diferentes** e combinar suas previsões.
Se cada árvore comete erros em lugares diferentes (porque são diversas), ao combinar
os votos, os erros tendem a se cancelar e os acertos se reforçam. Isso é chamado de
**ensemble** — o conjunto supera o indivíduo.

Para que isso funcione, as árvores precisam ser **diversas mas individualmente
competentes**. Não adianta combinar árvores idênticas (os erros seriam os mesmos),
nem árvores completamente aleatórias (cada uma erraria aleatoriamente, sem padrão
para se cancelar).

### Bootstrap: diversidade nos dados

Para cada árvore i, em vez de treinar com todos os 1200 exemplos de treino, sorteiam-se
1200 exemplos **com reposição**. Isso significa que alguns exemplos aparecem 2 ou 3 vezes
e outros não aparecem. Em média, cada bootstrap exclui ~37% dos exemplos originais
e duplica outros.

O efeito é que cada árvore aprende com uma "visão" ligeiramente diferente do dataset.
Um exemplo que é difícil de classificar pode estar presente em algumas árvores e
ausente em outras, gerando diferentes decisões de fronteira.

### Subset de features (max_features): diversidade na estrutura

Mesmo com bootstrap, se todas as árvores têm acesso às mesmas 3 features, o ID3
escolherá a melhor feature em cada nó — e essa escolha será quase sempre a mesma
entre as árvores (o ganho de informação é determinístico dado o conjunto de dados).
As árvores teriam estruturas muito parecidas e os votos seriam redundantes. Isso
é apenas **Bagging** (Bootstrap Aggregating), não Random Forest.

O Random Forest adiciona um segundo mecanismo: em cada nó de divisão, em vez de
avaliar as 3 features disponíveis, escolhem-se aleatoriamente **apenas 2**. A feature
vencedora é a melhor das 2 sorteadas, não necessariamente a melhor global. Isso faz
com que diferentes árvores construam divisões diferentes mesmo para os mesmos dados,
gerando diversidade estrutural real.

**Por que max_features = 2 especificamente?**

A regra padrão em Random Forest é `max_features = √n_features`. Com 3 features:

```
√3 ≈ 1.73 → arredondado para cima = 2
```

Com max_features=1, cada árvore usaria uma feature diferente em cada nó de forma
muito restrita — excesso de aleatoriedade, árvores pouco competentes.
Com max_features=3, todas as features são avaliadas — nenhuma aleatoriedade, degenera
em Bagging. O valor 2 é o ponto de equilíbrio entre competência e diversidade.

### Predição combinada

- **Classificação:** cada árvore emite um voto (classe 1–4). A classe com mais votos
  é a predição final. Com 50 árvores, um empate é raro e a classe dominante tende
  a ser a correta.

- **Regressão:** cada árvore estima uma gravidade (média da folha). A gravidade final
  é a **média** dessas 50 estimativas. Médias de múltiplos estimadores ruidosos tendem
  a convergir para o valor real (lei dos grandes números).

---

## 8. Escolha de N_ARVORES = 50

Testamos N_ARVORES ∈ {5, 10, 25, 50, 100} (N_FAIXAS=7, max_features=2):

| N_ARVORES | Acc Teste | RMSE Teste |
|-----------|-----------|------------|
| 5         | 79.67%    | 6.0773     |
| 10        | 83.00%    | 5.9999     |
| 25        | 82.67%    | 5.7961     |
| **50**    | **83.00%**| **5.7953** |
| 100       | 81.67%    | 5.7845     |

- N=50 empata com N=10 na acurácia e tem RMSE sensivelmente melhor
- N=100 piora a acurácia: com apenas 3 features, o subset aleatório às vezes
  força divisões ruins que se acumulam com muitas árvores
- N=50 escolhido como melhor custo-benefício entre acurácia e RMSE

---

## 9. Resultados finais (ID3 vs Random Forest)

| Modelo              | Acc Treino | Acc Teste | RMSE Treino | RMSE Teste |
|---------------------|------------|-----------|-------------|------------|
| ID3 simples         | 86.67%     | 82.33%    | 4.8808      | 5.9471     |
| Random Forest (N=50)| 86.50%     | 83.00%    | ~5.08       | 5.7953     |

O Random Forest melhora levemente a acurácia (+0.67%) e reduz o erro de regressão
(-0.15 RMSE). A melhora é modesta porque com apenas 3 features há um teto natural
para a diversidade entre as árvores.

---

## 10. Parâmetros finais

```
N_FAIXAS    = 7      # número de faixas por feature na discretização
N_ARVORES   = 50     # número de árvores no Random Forest
max_features = 2     # features avaliadas por divisão no RF (de 3 disponíveis)
pct_teste    = 0.20  # proporção do hold-out para teste
seed         = 42    # semente para reprodutibilidade do embaralhamento
```

---

## 11. Referências

### Árvores de Decisão — ID3, Entropia e Ganho de Informação

O algoritmo ID3 e o uso de entropia como critério de divisão são abordados em:

> RUSSELL, S. J.; NORVIG, P. **Inteligência Artificial: Uma Abordagem Moderna**.
> 3. ed. Rio de Janeiro: Elsevier, 2013.
> Cap. 18 — "Aprendizado a partir de Exemplos", seção 18.3 "Aprendizado de Árvores
> de Decisão". Cobre entropia, ganho de informação e a lógica de escolha do melhor
> atributo.

O tratamento mais completo e específico do ID3 (incluindo a derivação passo a passo
do algoritmo) está em:

> MITCHELL, T. M. **Machine Learning**. New York: McGraw-Hill, 1997.
> Cap. 3 — "Decision Tree Learning". Este capítulo foi escrito quando o ID3 era o
> estado da arte e detalha entropia, ganho de informação, overfitting em árvores e
> estratégias de poda.

O algoritmo ID3 foi proposto originalmente em:

> QUINLAN, J. R. Induction of Decision Trees. **Machine Learning**, v. 1, n. 1,
> p. 81–106, 1986.

---

### Discretização de Atributos Contínuos

Russell & Norvig e Mitchell não cobrem métodos de discretização em profundidade —
este é um tema de pré-processamento de dados. A referência mais completa para
equal-width, equal-frequency e suas comparações é:

> HAN, J.; KAMBER, M.; PEI, J. **Data Mining: Concepts and Techniques**. 3. ed.
> Waltham: Morgan Kaufmann, 2011.
> Cap. 3 — "Data Preprocessing", seção 3.4 "Data Discretization and Concept
> Hierarchy Generation". Compara equal-width, equal-frequency e discretização
> supervisionada.

---

### Ensemble, Bagging e Bootstrap

A técnica de bootstrap aggregating (Bagging) foi introduzida em:

> BREIMAN, L. Bagging Predictors. **Machine Learning**, v. 24, n. 2,
> p. 123–140, 1996.

Russell & Norvig cobre métodos de ensemble (incluindo bagging) em:

> RUSSELL, S. J.; NORVIG, P. **Inteligência Artificial: Uma Abordagem Moderna**.
> 3. ed. Cap. 18, seção 18.4 — "Métodos de Ensemble de Aprendizado". Explica
> votação, bagging e boosting de forma acessível.

---

### Random Forest

O Random Forest como combinação de bootstrap + subset aleatório de features foi
introduzido formalmente em:

> BREIMAN, L. Random Forests. **Machine Learning**, v. 45, n. 1, p. 5–32, 2001.

Este artigo é a referência primária para o algoritmo e justifica matematicamente
por que a diversidade das árvores (via subset de features) reduz o erro de
generalização. **Este conteúdo não está presente em Russell & Norvig com o mesmo
nível de detalhe** — o livro menciona florestas aleatórias brevemente mas não
desenvolve a justificativa do max_features.

---

### Avaliação de Modelos — Hold-out, Acurácia e RMSE

> RUSSELL, S. J.; NORVIG, P. **Inteligência Artificial: Uma Abordagem Moderna**.
> 3. ed. Cap. 18, seção 18.4 — "Teoria do Aprendizado". Cobre a divisão
> treino/teste e as métricas de avaliação de desempenho.

> MITCHELL, T. M. **Machine Learning**. Cap. 5 — "Evaluating Hypotheses".
> Detalha hold-out, validação cruzada e intervalos de confiança para acurácia.

