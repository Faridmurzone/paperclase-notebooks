import copy, json, os
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
chico = train[:100]
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")


def red(p=0.0):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Dropout(p),
                         nn.Linear(256, 256), nn.ReLU(), nn.Dropout(p), nn.Linear(256, 10))


def evaluar(modelo, idx):
    modelo.eval()
    with torch.no_grad():
        logits = modelo(X[idx])
        loss = F.cross_entropy(logits, y[idx]).item()
        acc = (logits.argmax(1) == y[idx]).float().mean().item()
    modelo.train()
    return loss, acc


def una_epoch(modelo, opt, epoch):
    orden = chico[torch.randperm(len(chico), generator=torch.Generator().manual_seed(epoch))]
    for i in range(0, len(orden), 25):
        b = orden[i:i + 25]
        opt.zero_grad()
        F.cross_entropy(modelo(X[b]), y[b]).backward()
        opt.step()

torch.manual_seed(0)
# >>>
modelo = nn.Sequential(
    nn.Linear(64, 256), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(256, 10))           # dropout después de cada ReLU

x = X[val[:1]]                    # una imagen
modelo.train()
a, b = modelo(x)[0, :3], modelo(x)[0, :3]
print("train 1:", a.detach().numpy().round(3))
print("train 2:", b.detach().numpy().round(3))
modelo.eval()
a, b = modelo(x)[0, :3], modelo(x)[0, :3]
print("eval 1 :", a.detach().numpy().round(3))
print("eval 2 :", b.detach().numpy().round(3))
# <<<
