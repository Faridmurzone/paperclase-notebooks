import json, math, os
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
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")


class Profunda(nn.Module):
    """20 hidden layers de 64. modo: plana, kaiming, residual, residual_ln."""
    def __init__(self, modo, capas=20, ancho=64):
        super().__init__()
        torch.manual_seed(0)
        self.modo = modo
        self.entrada = nn.Linear(64, ancho)
        self.capas = nn.ModuleList([nn.Linear(ancho, ancho) for _ in range(capas)])
        self.normas = nn.ModuleList([nn.LayerNorm(ancho) for _ in range(capas)])
        self.salida = nn.Linear(ancho, 10)
        if modo == "kaiming":
            for c in self.capas:
                nn.init.kaiming_normal_(c.weight, nonlinearity="relu")
                nn.init.zeros_(c.bias)

    def forward(self, x):
        h = self.entrada(x)
        for capa, norma in zip(self.capas, self.normas):
            if self.modo == "residual":
                h = h + torch.relu(capa(h))
            elif self.modo == "residual_ln":
                h = h + torch.relu(capa(norma(h)))
            else:
                h = torch.relu(capa(h))
        return self.salida(h)


def grads_por_capa(modelo):
    loss = F.cross_entropy(modelo(X[train[:256]]), y[train[:256]])
    loss.backward()
    return [c.weight.grad.norm().item() for c in modelo.capas]



def entrenar(modelo, lr=0.05, epochs=5):
    opt = torch.optim.SGD(modelo.parameters(), lr=lr)
    for e in range(1, epochs + 1):
        o = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(e))]
        for i in range(0, len(o), 32):
            b = o[i:i + 32]
            opt.zero_grad()
            F.cross_entropy(modelo(X[b]), y[b]).backward()
            opt.step()
    modelo.eval()
    with torch.no_grad():
        lg = modelo(X[val])
        return F.cross_entropy(lg, y[val]).item(), (lg.argmax(1) == y[val]).float().mean().item()
# >>>
# 20 hidden layers, 5 epochs, SGD con learning rate 0,05
for modo in ("plana", "kaiming", "residual", "residual_ln"):
    lv, acc = entrenar(Profunda(modo))
    print(f"{modo:12} val {lv:.3f}  acc {acc:.3f}")
# <<<
