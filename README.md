# sistemas_inteligentes

trabalho 1: busca local com tempera simulada e algoritmo genetico linguagem C.

```bash
make run-gen
```

exemplo de saida:
```
vetor [ 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 ] (valor = 197 peso = 169)
vetor [ 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 ] (valor = 197 peso = 169)
vetor [ 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 ] (valor = 174 peso = 114)
vetor [ 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 ] (valor = 197 peso = 169)
vetor [ 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 ] (valor = 126 peso = 241)
vetor [ 0 0 1 0 0 1 0 0 0 0 0 0 1 0 1 ] (valor = 227 peso = 207)
vetor [ 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 ] (valor = 197 peso = 169)
vetor [ 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 ] (valor = 197 peso = 169)
vetor [ 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 ] (valor = 114 peso = 125)
vetor [ 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 ] (valor = 181 peso = 277)
geracao: 34 nivel de estagnacao: 19
itens[15] { (27, 42) (7, 163) (23, 55) (43, 193) (74, 170) (91, 70) (65, 88) (18, 70) (35, 171) (11, 169) (78, 175) (22, 43) (30, 38) (12, 108) (83, 44) } W = 200
best fitness: vetor [ 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 ] (valor = 197 peso = 169)
```

```bash
make run-temp
```
exemplo de saida:
```
.....
vetor [ 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 ] (valor = 209 peso = 191)
vetor [ 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 ] (valor = 209 peso = 191)
vetor [ 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 ] (valor = 209 peso = 191)
vetor [ 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 ] (valor = 209 peso = 191)
vetor [ 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 ] (valor = 209 peso = 191)
vetor [ 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 ] (valor = 209 peso = 191)
vetor [ 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 ] (valor = 209 peso = 191)
vetor [ 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 ] (valor = 209 peso = 191)
vetor [ 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 ] (valor = 209 peso = 191)
vetor [ 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 ] (valor = 209 peso = 191)
vetor [ 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 ] (valor = 209 peso = 191)
vetor [ 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 ] (valor = 209 peso = 191)
vetor [ 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 ] (valor = 209 peso = 191)
melhor estado
vetor [ 0 0 0 0 0 1 0 1 0 0 1 0 0 0 0 ] (valor = 209 peso = 191)
itens[15] { (64, 119) (13, 198) (23, 128) (16, 88) (44, 57) (95, 21) (80, 154) (66, 97) (58, 172) (27, 149) (48, 73) (82, 127) (69, 175) (57, 158) (18, 32) } W = 200
```

trabalho 2: classificacao e regressao com Random Forest com ID3 e MLP em python.

```bash
python id3.py
```
saida:
```
=== ID3 ===
           Acuracia      RMSE
Treino       86.67%    4.8808
Teste        82.33%    5.9471

=== Random Forest (N_ARVORES=50, max_features=2) ===
           Acuracia      RMSE
Treino       86.50%    5.0054
Teste        83.33%    5.7931

(N_FAIXAS=7, N_ARVORES=50)
```

```bash
python mlp.py
```
saida:
```
python mlp.py
=== MLP (H_SIZE=6, EPOCAS=200, TAXA=0.01) ===
           Acuracia      RMSE
Treino       85.67%    4.3098
Teste        84.33%    4.4455
```
