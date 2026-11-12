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

modelo = red()
opt = torch.optim.SGD(modelo.parameters(), lr=0.05, momentum=0.9)
# >>>
paciencia = 20                    # epochs sin mejorar que toleramos
mejor_loss, mejor_epoch = float("inf"), 0
for epoch in range(1, 201):
    una_epoch(modelo, opt, epoch)
    lv, _ = evaluar(modelo, val)
    if lv < mejor_loss:           # mejoró: guardar un checkpoint
        mejor_loss, mejor_epoch = lv, epoch
        mejor = copy.deepcopy(modelo.state_dict())
    elif epoch - mejor_epoch >= paciencia:
        break                     # 20 epochs sin mejorar: cortar

modelo.load_state_dict(mejor)     # volver al mejor
print("cortó en la epoch", epoch, "· mejor:", mejor_epoch)
lv, av = evaluar(modelo, val)
print(f"val restaurado: pérdida {lv:.3f} acc {av:.3f}")
# <<<
