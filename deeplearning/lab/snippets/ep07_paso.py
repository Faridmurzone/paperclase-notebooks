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
torch.manual_seed(0)
modelo = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
opt = torch.optim.SGD(modelo.parameters(), lr=0.05, momentum=0.9)
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")

# >>>
opt = torch.optim.SGD(modelo.parameters(), lr=0.05, momentum=0.9)
b = train[:25]                             # un batch de 25
for paso in range(4):
    opt.zero_grad()                        # 1. borrar los .grad
    logits = modelo(X[b])                  # 2. forward
    loss = F.cross_entropy(logits, y[b])   # 3. pérdida
    loss.backward()                        # 4. backward: llena .grad
    opt.step()                             # 5. mover los pesos
    print(f"paso {paso}: pérdida {loss.item():.3f}")

# ¿y sin zero_grad? los .grad se suman
w = modelo[0].weight
opt.zero_grad(); F.cross_entropy(modelo(X[b]), y[b]).backward()
g1 = w.grad.norm().item()
F.cross_entropy(modelo(X[b]), y[b]).backward()   # segunda vez
print(f"norma .grad: {g1:.3f} → {w.grad.norm().item():.3f}")
# <<<
