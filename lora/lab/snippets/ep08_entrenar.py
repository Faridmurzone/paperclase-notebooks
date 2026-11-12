import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
torch.manual_seed(0)
N = torch.distributions.Normal(0, 1)
_neg = N.icdf(torch.linspace(0.03, 0.5, 8))
_pos = N.icdf(torch.linspace(0.5, 0.97, 9))
NF4 = torch.cat([_neg, _pos[1:]])
NF4 = NF4 / NF4.abs().max()


class Lineal4bit(nn.Module):
    def __init__(self, lin, r=8, alpha=16, bloque=64):
        super().__init__()
        w = lin.weight.detach().reshape(-1, bloque)
        absmax = w.abs().max(dim=1, keepdim=True).values
        codigo = ((w / absmax)[..., None] - NF4).abs().argmin(-1)
        self.register_buffer("codigo", codigo.to(torch.uint8))
        self.register_buffer("absmax", absmax)
        self.register_buffer("bias", lin.bias.detach().clone())
        self.forma = lin.weight.shape
        self.A = nn.Parameter(torch.randn(r, self.forma[1]) * 0.01)
        self.B = nn.Parameter(torch.zeros(self.forma[0], r))
        self.escala = alpha / r

    def forward(self, x):
        W = (NF4[self.codigo.long()] * self.absmax).reshape(self.forma)  # dequant
        lora = x @ self.A.T @ self.B.T            # el desvío, en float32
        return F.linear(x, W, self.bias) + self.escala * lora


class LoRALinear(nn.Module):                    # LoRA sobre la base en float32
    def __init__(self, lin, r=8, alpha=16):
        super().__init__()
        self.base = lin.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, lin.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(lin.out_features, r))
        self.escala = alpha / r

    def forward(self, x):
        return self.base(x) + self.escala * (x @ self.A.T @ self.B.T)


# dígitos 8×8 de sklearn; la tarea nueva: los mismos dígitos en espejo
X, y = load_digits(return_X_y=True)
X, y = torch.tensor(X, dtype=torch.float32) / 16, torch.tensor(y)
espejo = X.view(-1, 8, 8).flip(2).reshape(-1, 64)
orden = torch.randperm(len(X))
tr, te = orden[:1200], orden[1200:]
base = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
opt = torch.optim.Adam(base.parameters(), 1e-3)
for _ in range(300):
    perdida = F.cross_entropy(base(X[tr]), y[tr])
    opt.zero_grad()
    perdida.backward()
    opt.step()


def acierto(m, Z):
    with torch.no_grad():
        return (m(Z[te]).argmax(1) == y[te]).float().mean().item()


def con_lora(Capa):
    torch.manual_seed(1)
    m = copy.deepcopy(base)
    for i in [0, 2, 4]:
        m[i] = Capa(m[i])
    return m


def entrenar(m, Z, pasos=150):
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], 1e-2)
    for _ in range(pasos):
        perdida = F.cross_entropy(m(Z[tr]), y[tr])
        opt.zero_grad()
        perdida.backward()
        opt.step()


# >>>
print("base      dígitos → espejo")
print(f"sin LoRA:  {acierto(base, X):.1%}  →  {acierto(base, espejo):.1%}")

for nombre, Capa in [("float32", LoRALinear), ("NF4", Lineal4bit)]:
    m = con_lora(Capa)                       # A al azar, B = 0
    antes = acierto(m, X)                    # ¿cuánto perdió la base?
    entrenar(m, espejo)                      # solo A y B
    print(f"{nombre + ':':9}  {antes:.1%}  →  {acierto(m, espejo):.1%}")
n = sum(p.numel() for p in m.parameters() if p.requires_grad)
total = sum(p.numel() for p in base.parameters())
print(f"entrenables {n:,} (base: {total:,})")
# <<<
