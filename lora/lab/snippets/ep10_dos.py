import copy
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
torch.manual_seed(0)
# tarea vieja: dígitos de sklearn (8×8); tarea nueva: los mismos dígitos en espejo
X, y = load_digits(return_X_y=True)
X, y = torch.tensor(X, dtype=torch.float32) / 16, torch.tensor(y)
espejo = X.view(-1, 8, 8).flip(2).reshape(-1, 64)
orden = torch.randperm(len(X))
tr, te = orden[:1200], orden[1200:]
base = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
_opt = torch.optim.Adam(base.parameters(), 1e-3)
for _ in range(300):
    _l = F.cross_entropy(base(X[tr]), y[tr])
    _opt.zero_grad()
    _l.backward()
    _opt.step()
W0 = base[2].weight.detach().clone()          # la capa del medio, 256 × 256


class LoRALinear(nn.Module):                  # la del episodio cinco
    def __init__(self, lin, r, alpha):
        super().__init__()
        self.base = lin.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, lin.in_features) / lin.in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(lin.out_features, r))
        self.escala = alpha / r

    def forward(self, x):
        return self.base(x) + self.escala * (x @ self.A.T @ self.B.T)

    def delta(self):
        return (self.escala * self.B @ self.A).detach()


def acierto(m, Z):
    with torch.no_grad():
        return (m(Z[te]).argmax(1) == y[te]).float().mean().item()


def entrenar(m, Z, lr, pasos=200):
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr)
    for _ in range(pasos):
        perdida = F.cross_entropy(m(Z[tr]), y[tr])
        opt.zero_grad()
        perdida.backward()
        opt.step()


def completo(lr):
    m = copy.deepcopy(base)
    entrenar(m, espejo, lr)
    return m


def con_lora(r, lr):
    torch.manual_seed(1)
    m = copy.deepcopy(base)
    for i in [0, 2, 4]:
        m[i] = LoRALinear(m[i], r=r, alpha=2 * r)
    entrenar(m, espejo, lr)
    return m


def delta(m):                                  # ΔW de la capa del medio
    return m[2].delta() if isinstance(m[2], LoRALinear) else m[2].weight.detach() - W0


def rango90(D):                                # cuántos σ juntan el 90 % de la energía
    S = torch.linalg.svdvals(D)
    energia = (S ** 2).cumsum(0) / (S ** 2).sum()
    return int((energia < 0.9).sum()) + 1


# >>>
ft = copy.deepcopy(base)                     # fine-tuning completo
entrenar(ft, espejo, lr=1e-3)                # todos los pesos

torch.manual_seed(1)                         # la misma semilla en todo el episodio
lo = copy.deepcopy(base)                     # LoRA de rango 16
for i in [0, 2, 4]:
    lo[i] = LoRALinear(lo[i], r=16, alpha=32)
entrenar(lo, espejo, lr=1e-3)                # solo A y B

print("             espejo  dígitos")
for nombre, m in [("base", base), ("completo", ft), ("LoRA r=16", lo)]:
    print(f"{nombre:10} {acierto(m, espejo):8.1%} {acierto(m, X):8.1%}")
for nombre, m in [("completo", ft), ("LoRA r=16", lo)]:
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"entrenan en {nombre}: {n:,}")
# <<<
