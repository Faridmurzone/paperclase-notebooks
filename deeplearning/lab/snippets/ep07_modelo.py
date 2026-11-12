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
modelo = nn.Sequential(
    nn.Linear(64, 256), nn.ReLU(),     # 64 píxeles → 256
    nn.Linear(256, 256), nn.ReLU(),    # hidden layer
    nn.Linear(256, 10))                # un logit por dígito
n = sum(p.numel() for p in modelo.parameters())
print("parámetros:", n)

logits = modelo(X[train[:4]])          # un batch de 4 imágenes
por_ejemplo = F.cross_entropy(logits, y[train[:4]], reduction="none")
print("logits:", tuple(logits.shape))
print("por ejemplo:", por_ejemplo.detach().numpy().round(3))
print("promedio:", round(por_ejemplo.mean().item(), 3),
      "  ln(10) =", round(math.log(10), 3))
# <<<
