"""Modelo y datos de juguete compartidos por los snippets de la serie LoRA (no es un snippet: no empieza con epNN)."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits

DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
X, y = load_digits(return_X_y=True)                     # 1797 dígitos de 8×8, sintéticos de sklearn
X = torch.tensor(X / 16, dtype=torch.float32)
y = torch.tensor(y)
_perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
Xtr, ytr, Xte, yte = X[_perm[:1400]], y[_perm[:1400]], X[_perm[1400:]], y[_perm[1400:]]


def espejo(X):
    """El mismo dígito reflejado de izquierda a derecha: la tarea nueva."""
    return X.view(-1, 8, 8).flip(2).reshape(-1, 64)


def red_nueva():
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))


def precision(red, X, y):
    with torch.no_grad():
        return (red(X).argmax(1) == y).float().mean().item()


def entrenar(red, X, y, pasos, lr=1e-3, params=None, semilla=0):
    g = torch.Generator().manual_seed(semilla)
    opt = torch.optim.Adam(params if params is not None else red.parameters(), lr=lr)
    for _ in range(pasos):
        i = torch.randint(0, len(X), (64,), generator=g)
        perdida = F.cross_entropy(red(X[i]), y[i])
        opt.zero_grad()
        perdida.backward()
        opt.step()
    return red


def cargar_base():
    red = red_nueva()
    red.load_state_dict(torch.load(os.path.join(DATOS, "ep02_base.pt")))
    return red


class LoraMini(nn.Module):
    """LoRA mínima para los experimentos del episodio 4 (la versión completa y verificada es la del episodio 5)."""

    def __init__(self, base, r, alfa=8):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features**0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.s = alfa / r

    def forward(self, x):
        return self.base(x) + self.s * (x @ self.A.T) @ self.B.T


def con_lora(capas, r, pasos=300, lr=1e-2):
    """Red base congelada + LoRA de rango r en las capas indicadas, entrenada en espejo. Devuelve (entrenables, precisión)."""
    torch.manual_seed(0)
    red = cargar_base().requires_grad_(False)
    for i in capas:
        red[i] = LoraMini(red[i], r)
    ps = [p for p in red.parameters() if p.requires_grad]
    entrenar(red, espejo(Xtr), ytr, pasos=pasos, lr=lr, params=ps)
    return sum(p.numel() for p in ps), precision(red, espejo(Xte), yte)


class LoRALinear(nn.Module):
    """La capa LoRA del episodio 5 (ep05_clase.py la muestra y verifica que es idéntica a esta)."""

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)          # W0 y b: congelados
        k, d = base.in_features, base.out_features
        self.A = nn.Parameter(torch.randn(r, k) / k**0.5)  # r × k, al azar
        self.B = nn.Parameter(torch.zeros(d, r))        # d × r, en cero
        self.escala = alpha / r

    def forward(self, x):
        return self.base(x) + self.escala * (x @ self.A.T @ self.B.T)


def agregar_lora(red, r, alpha):
    """Congela toda la red y reemplaza cada nn.Linear de un nn.Sequential por una LoRALinear."""
    red.requires_grad_(False)
    for i, capa in enumerate(red):
        if isinstance(capa, nn.Linear):
            red[i] = LoRALinear(capa, r, alpha)
    return red
