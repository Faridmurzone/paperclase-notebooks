import json, os
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.datasets import load_digits
torch.set_num_threads(1)
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16, dtype=torch.float32); y = torch.tensor(y)
perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
train, val = perm[:1200], perm[1200:]

# >>>
def entrenar(B, lr=0.1, epochs=10):
    torch.manual_seed(0)
    modelo = nn.Linear(64, 10)                 # regresión logística
    opt = torch.optim.SGD(modelo.parameters(), lr=lr)
    hist, pasos = [], 0
    for epoch in range(epochs):
        orden = train[torch.randperm(len(train))]
        for i in range(0, len(orden), B):      # recorrer en minibatches
            b = orden[i:i + B]
            opt.zero_grad()
            F.cross_entropy(modelo(X[b]), y[b]).backward()
            opt.step(); pasos += 1
        with torch.no_grad():
            hist.append(F.cross_entropy(modelo(X[train]), y[train]).item())
    return modelo, hist, pasos

for B in [1200, 32, 1]:
    modelo, hist, pasos = entrenar(B)
    acc = (modelo(X[val]).argmax(1) == y[val]).float().mean().item()
    print(f"B = {B:4}: {pasos:5} pasos  pérdida {hist[-1]:.3f}  acc val {acc:.3f}")
# <<<
json.dump({str(B): entrenar(B)[1] for B in [1200, 32, 1]}, open(os.path.join(DATOS, "ep02_digitos.json"), "w"))
