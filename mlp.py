import math
import random

FEATURE_RANGES = {
    'qPA':   (-10.0, 10.0),
    'pulso': (0.0, 200.0),
    'resp':  (0.0, 22.0),
}

H_SIZE    = 6
TAXA      = 0.01
EPOCAS    = 200
SEED      = 42
MLP_SEED  = 5


def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def normaliza(qpa, pulso, resp):
    def norm(v, vmin, vmax):
        return (v - vmin) / (vmax - vmin)
    return [
        norm(qpa,   *FEATURE_RANGES['qPA']),
        norm(pulso, *FEATURE_RANGES['pulso']),
        norm(resp,  *FEATURE_RANGES['resp']),
    ]


def load_labeled(filepath):
    X, gravidades, labels = [], [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [x.strip() for x in line.split(',')]
            qpa, pulso, resp = float(parts[3]), float(parts[4]), float(parts[5])
            X.append(normaliza(qpa, pulso, resp))
            gravidades.append(float(parts[6]))
            labels.append(int(parts[7]))
    return X, gravidades, labels


def hold_out(X, gravidades, labels, pct_teste=0.2, seed=SEED):
    random.seed(seed)
    dados = list(zip(X, gravidades, labels))
    random.shuffle(dados)
    corte = int(len(dados) * (1 - pct_teste))
    X_tr, g_tr, l_tr = zip(*dados[:corte])
    X_te, g_te, l_te = zip(*dados[corte:])
    return list(X_tr), list(g_tr), list(l_tr), list(X_te), list(g_te), list(l_te)


class Neuronio:
    def __init__(self, n_entradas, ativacao='sigmoid'):
        self.pesos    = [random.uniform(-0.5, 0.5) for _ in range(n_entradas + 1)]
        self.ativacao = ativacao
        self.saida    = 0.0
        self.delta    = 0.0
        self.entradas = []

    def ativar(self, entradas):
        self.entradas = entradas + [1.0]
        net = sum(p * x for p, x in zip(self.pesos, self.entradas))
        self.saida = sigmoid(net) if self.ativacao == 'sigmoid' else net
        return self.saida

    def atualizar(self, taxa):
        for i in range(len(self.pesos)):
            self.pesos[i] -= taxa * self.delta * self.entradas[i]


class MLP:
    def __init__(self, n_entradas=3, h_size=H_SIZE, seed=SEED):
        random.seed(seed)
        self.oculta = [Neuronio(n_entradas, 'sigmoid') for _ in range(h_size)]
        self.saida  = Neuronio(h_size, 'linear')

    def forward(self, x):
        ativ = [n.ativar(x) for n in self.oculta]
        return self.saida.ativar(ativ)

    def backprop(self, x, y_true, taxa):
        y_pred = self.forward(x)

        self.saida.delta = y_pred - y_true

        for j, n in enumerate(self.oculta):
            n.delta = n.saida * (1 - n.saida) * self.saida.pesos[j] * self.saida.delta

        self.saida.atualizar(taxa)
        for n in self.oculta:
            n.atualizar(taxa)

    def treinar_sgd(self, X, gravidades, epocas=EPOCAS, taxa=TAXA, seed=SEED):
        random.seed(seed)
        ordem = list(range(len(X)))
        for _ in range(epocas):
            random.shuffle(ordem)
            for i in ordem:
                self.backprop(X[i], gravidades[i], taxa)

    def treinar_batch(self, X, gravidades, epocas=EPOCAS, taxa=TAXA):
        n = len(X)
        for _ in range(epocas):
            grads_saida  = [0.0] * len(self.saida.pesos)
            grads_oculta = [[0.0] * len(nj.pesos) for nj in self.oculta]
            for i in range(n):
                self.forward(X[i])
                delta_s = self.saida.saida - gravidades[i]
                for k in range(len(grads_saida)):
                    grads_saida[k] += delta_s * self.saida.entradas[k]
                for j, nj in enumerate(self.oculta):
                    d_j = nj.saida * (1 - nj.saida) * self.saida.pesos[j] * delta_s
                    for k in range(len(grads_oculta[j])):
                        grads_oculta[j][k] += d_j * nj.entradas[k]
            for k in range(len(self.saida.pesos)):
                self.saida.pesos[k] -= taxa * grads_saida[k] / n
            for j, nj in enumerate(self.oculta):
                for k in range(len(nj.pesos)):
                    nj.pesos[k] -= taxa * grads_oculta[j][k] / n


def inferir_centroids(gravidades, labels):
    grupos = {}
    for g, l in zip(gravidades, labels):
        grupos.setdefault(l, []).append(g)
    return {l: sum(v) / len(v) for l, v in grupos.items()}


def gravidade_para_classe(g, centroids):
    return min(centroids, key=lambda c: abs(centroids[c] - g))


def avalia(mlp, X, gravidades, labels, centroids):
    preds_g = [mlp.forward(x) for x in X]
    preds_c = [gravidade_para_classe(g, centroids) for g in preds_g]
    acc  = sum(pc == lc for pc, lc in zip(preds_c, labels)) / len(labels)
    rmse = math.sqrt(sum((pg - gr) ** 2 for pg, gr in zip(preds_g, gravidades)) / len(gravidades))
    return acc, rmse


if __name__ == "__main__":
    LABELED = "02_treino_sinais_vitais_com_label.txt"

    X, gravidades, labels = load_labeled(LABELED)
    X_tr, g_tr, l_tr, X_te, g_te, l_te = hold_out(X, gravidades, labels)
    centroids = inferir_centroids(g_tr, l_tr)

    mlp = MLP(h_size=H_SIZE, seed=MLP_SEED)
    mlp.treinar_sgd(X_tr, g_tr, seed=MLP_SEED)

    a_tr, r_tr = avalia(mlp, X_tr, g_tr, l_tr, centroids)
    a_te, r_te = avalia(mlp, X_te, g_te, l_te, centroids)

    print(f"=== MLP (H_SIZE={H_SIZE}, EPOCAS={EPOCAS}, TAXA={TAXA}) ===")
    print(f"{'':8} {'Acuracia':>10}  {'RMSE':>8}")
    print(f"{'Treino':<8} {a_tr:>10.2%}  {r_tr:>8.4f}")
    print(f"{'Teste':<8} {a_te:>10.2%}  {r_te:>8.4f}")
