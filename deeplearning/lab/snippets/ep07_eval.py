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
def evaluar(idx):
    modelo.eval()                  # modo evaluación
    with torch.no_grad():          # sin grafo: no vamos a derivar
        logits = modelo(X[idx])
        loss = F.cross_entropy(logits, y[idx]).item()
        pred = logits.argmax(1)    # el logit más grande gana
        acc = (pred == y[idx]).float().mean().item()
    modelo.train()                 # de vuelta a modo entrenamiento
    return loss, acc

loss, acc = evaluar(val)
print(f"val sin entrenar: pérdida {loss:.3f} acc {acc:.3f}")
# <<<
