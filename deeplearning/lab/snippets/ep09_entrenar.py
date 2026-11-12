import torch
import torch.nn.functional as F
from torch import nn
from sklearn.datasets import load_digits
torch.set_num_threads(1)
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16, dtype=torch.float32)
y = torch.tensor(y)
perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
train, val = perm[:1200], perm[1200:]


def evaluar(m):
    m.eval()
    with torch.no_grad():
        lg = m(X[val])
        r = F.cross_entropy(lg, y[val]).item(), (lg.argmax(1) == y[val]).float().mean().item()
    m.train()
    return r


def entrenar(m, epochs=5):
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    for e in range(1, epochs + 1):
        o = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(e))]
        for i in range(0, len(o), 32):
            b = o[i:i + 32]
            opt.zero_grad()
            F.cross_entropy(m(X[b]), y[b]).backward()
            opt.step()
    return evaluar(m)
# >>>
def red(norm=None, capas=6, ancho=64):
    torch.manual_seed(0)
    bloques, d = [], 64
    for _ in range(capas):                   # 6 hidden layers
        bloques.append(nn.Linear(d, ancho))
        if norm is not None:
            bloques.append(norm(ancho))      # normalizar antes de ReLU
        bloques.append(nn.ReLU())
        d = ancho
    return nn.Sequential(*bloques, nn.Linear(d, 10))

for nombre, norm in [("sin norma", None), ("BatchNorm", nn.BatchNorm1d),
                     ("LayerNorm", nn.LayerNorm)]:
    lv, acc = entrenar(red(norm))            # 5 epochs, SGD lr=0,1
    print(f"{nombre:10} val {lv:.3f}  acc {acc:.3f}")
# <<<
