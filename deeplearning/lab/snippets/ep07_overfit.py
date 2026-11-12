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

def evaluar(idx):
    modelo.eval()
    with torch.no_grad():
        logits = modelo(X[idx])
        loss = F.cross_entropy(logits, y[idx]).item()
        acc = (logits.argmax(1) == y[idx]).float().mean().item()
    modelo.train()
    return loss, acc

def entrenar_una_epoch(idx, epoch, tam=25):
    orden = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(epoch))]
    for i in range(0, len(orden), tam):
        b = orden[i:i + tam]
        opt.zero_grad()
        F.cross_entropy(modelo(X[b]), y[b]).backward()
        opt.step()

# >>>
chico = train[:100]                   # solo 100 ejemplos
historia = []
for epoch in range(1, 201):
    entrenar_una_epoch(chico, epoch)  # el mismo loop de antes
    lt, _ = evaluar(chico)
    lv, av = evaluar(val)
    historia.append((epoch, lt, lv, av))

mejor = min(historia, key=lambda h: h[2])   # menor pérdida en val
for e, lt, lv, av in (historia[0], mejor, historia[-1]):
    print(f"ep {e:3}: train {lt:.3f} val {lv:.3f} acc {av:.3f}")
# <<<
json.dump(historia, open(os.path.join(DATOS, "ep07_curvas.json"), "w"))
