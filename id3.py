import math
from collections import Counter


FEATURE_RANGES = {
    'qPA':   (-10.0, 10.0),
    'pulso': (0.0, 200.0),
    'resp':  (0.0, 22.0),
}

N_FAIXAS      = 7
N_ARVORES     = 50
SEED          = 42
RF_SEED       = 1


def discretize(value, vmin, vmax, n_faixas):
    if value <= vmin:
        return 0
    if value >= vmax:
        return n_faixas - 1
    largura = (vmax - vmin) / n_faixas
    return int((value - vmin) / largura)


def discretize_exemplo(qpa, pulso, resp, n_faixas=N_FAIXAS):
    return [
        discretize(qpa,   *FEATURE_RANGES['qPA'],   n_faixas),
        discretize(pulso, *FEATURE_RANGES['pulso'], n_faixas),
        discretize(resp,  *FEATURE_RANGES['resp'],  n_faixas),
    ]


def load_labeled(filepath, n_faixas=N_FAIXAS):
    examples, labels, gravities = [], [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [x.strip() for x in line.split(',')]
            qpa, pulso, resp = float(parts[3]), float(parts[4]), float(parts[5])
            examples.append(discretize_exemplo(qpa, pulso, resp, n_faixas))
            gravities.append(float(parts[6]))
            labels.append(int(parts[7]))
    return examples, labels, gravities


def load_test(filepath, n_faixas=N_FAIXAS):
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [x.strip() for x in line.split(',')]
            idx  = int(parts[0])
            qpa, pulso, resp = float(parts[3]), float(parts[4]), float(parts[5])
            rows.append((idx, discretize_exemplo(qpa, pulso, resp, n_faixas)))
    return rows


class Node:
    def __init__(self):
        self.feature  = None
        self.children = {}
        self.label    = None
        self.gravity  = None

    def is_leaf(self):
        return self.label is not None


def entropy(labels):
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def information_gain(examples, feature_idx, labels):
    total = len(examples)
    subsets = {}
    for ex, lbl in zip(examples, labels):
        subsets.setdefault(ex[feature_idx], []).append(lbl)
    weighted = sum((len(v) / total) * entropy(v) for v in subsets.values())
    return entropy(labels) - weighted


def majority_class(labels):
    return Counter(labels).most_common(1)[0][0]


def id3(examples, labels, gravities, features, max_features=None):
    node = Node()

    node.gravity = sum(gravities) / len(gravities)

    if len(set(labels)) == 1:
        node.label = labels[0]
        return node

    if not features:
        node.label = majority_class(labels)
        return node

    import random
    candidatos = features
    if max_features and len(features) > max_features:
        candidatos = set(random.sample(list(features), max_features))

    best = max(candidatos, key=lambda f: information_gain(examples, f, labels))
    node.feature = best
    remaining = features - {best}

    for val in set(ex[best] for ex in examples):
        sub_ex  = [ex  for ex, lbl, g in zip(examples, labels, gravities) if ex[best] == val]
        sub_lbl = [lbl for ex, lbl, g in zip(examples, labels, gravities) if ex[best] == val]
        sub_g   = [g   for ex, lbl, g in zip(examples, labels, gravities) if ex[best] == val]
        node.children[val] = id3(sub_ex, sub_lbl, sub_g, remaining, max_features)

    return node


def _collect_leaf_data(node):
    if node.is_leaf():
        return [(node.label, node.gravity)]
    return [item for child in node.children.values() for item in _collect_leaf_data(child)]


def predict(node, example):
    if node.is_leaf():
        return node.label, node.gravity
    val = example[node.feature]
    if val not in node.children:
        data     = _collect_leaf_data(node)
        labels   = [d[0] for d in data]
        gravs    = [d[1] for d in data]
        return majority_class(labels), sum(gravs) / len(gravs)
    return predict(node.children[val], example)


def print_tree(node, feature_names=None, depth=0, branch_val=None):
    indent = "  " * depth
    prefix = f"[{branch_val}] " if branch_val is not None else ""
    if node.is_leaf():
        print(f"{indent}{prefix}-> classe {node.label}  (grav_media={node.gravity:.2f})")
        return
    fname = feature_names[node.feature] if feature_names else f"f{node.feature}"
    print(f"{indent}{prefix}? {fname}")
    for val, child in sorted(node.children.items()):
        print_tree(child, feature_names, depth + 1, val)


def hold_out(examples, labels, gravities, pct_teste=0.2, seed=SEED):
    import random
    random.seed(seed)
    dados = list(zip(examples, labels, gravities))
    random.shuffle(dados)
    corte = int(len(dados) * (1 - pct_teste))
    treino = dados[:corte]
    teste  = dados[corte:]
    ex_tr, lbl_tr, grav_tr = zip(*treino)
    ex_te, lbl_te, grav_te = zip(*teste)
    return list(ex_tr), list(lbl_tr), list(grav_tr), list(ex_te), list(lbl_te), list(grav_te)


def bootstrap(examples, labels, gravities, seed):
    import random
    random.seed(seed)
    n       = len(examples)
    indices = [random.randint(0, n - 1) for _ in range(n)]
    return (
        [examples[i]  for i in indices],
        [labels[i]    for i in indices],
        [gravities[i] for i in indices],
    )


class RandomForest:
    def __init__(self, n_arvores=N_ARVORES, max_features=2):
        self.n_arvores    = n_arvores
        self.max_features = max_features
        self.trees        = []

    def fit(self, examples, labels, gravities):
        self.trees = []
        for i in range(self.n_arvores):
            ex_b, lbl_b, grav_b = bootstrap(examples, labels, gravities, seed=RF_SEED + i)
            self.trees.append(id3(ex_b, lbl_b, grav_b, {0, 1, 2}, self.max_features))

    def predict(self, example):
        resultados = [predict(tree, example) for tree in self.trees]
        votos      = [c for c, _ in resultados]
        grav_media = sum(g for _, g in resultados) / len(resultados)
        return majority_class(votos), grav_media

    def predict_all(self, examples):
        return [self.predict(ex) for ex in examples]


def avalia(preds, lbl_real, grav_real):
    acc  = sum(c == lbl for (c, _), lbl in zip(preds, lbl_real)) / len(lbl_real)
    rmse = math.sqrt(sum((g - gr)**2 for (_, g), gr in zip(preds, grav_real)) / len(grav_real))
    return acc, rmse


if __name__ == "__main__":
    LABELED = "02_treino_sinais_vitais_com_label.txt"

    examples, labels, gravities = load_labeled(LABELED)
    ex_tr, lbl_tr, grav_tr, ex_te, lbl_te, grav_te = hold_out(examples, labels, gravities)

    print("=== ID3 ===")
    tree       = id3(ex_tr, lbl_tr, grav_tr, {0, 1, 2})
    a_tr, r_tr = avalia([predict(tree, ex) for ex in ex_tr], lbl_tr, grav_tr)
    a_te, r_te = avalia([predict(tree, ex) for ex in ex_te], lbl_te, grav_te)
    print(f"{'':8} {'Acuracia':>10}  {'RMSE':>8}")
    print(f"{'Treino':<8} {a_tr:>10.2%}  {r_tr:>8.4f}")
    print(f"{'Teste':<8} {a_te:>10.2%}  {r_te:>8.4f}")

    print(f"\n=== Random Forest (N_ARVORES={N_ARVORES}, max_features=2) ===")
    rf = RandomForest()
    rf.fit(ex_tr, lbl_tr, grav_tr)
    fa_tr, fr_tr = avalia(rf.predict_all(ex_tr), lbl_tr, grav_tr)
    fa_te, fr_te = avalia(rf.predict_all(ex_te), lbl_te, grav_te)
    print(f"{'':8} {'Acuracia':>10}  {'RMSE':>8}")
    print(f"{'Treino':<8} {fa_tr:>10.2%}  {fr_tr:>8.4f}")
    print(f"{'Teste':<8} {fa_te:>10.2%}  {fr_te:>8.4f}")

    print(f"\n(N_FAIXAS={N_FAIXAS}, N_ARVORES={N_ARVORES})")
