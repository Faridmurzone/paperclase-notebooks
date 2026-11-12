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

def entrenar(p=0.0, wd=0.0):
    modelo = red(p)
    opt = torch.optim.SGD(modelo.parameters(), lr=0.05, momentum=0.9, weight_decay=wd)
    historia = []
    for epoch in range(1, 201):
        una_epoch(modelo, opt, epoch)
        historia.append((epoch, *evaluar(modelo, val)))
    return modelo, historia


def norma(modelo):
    return sum(w.pow(2).sum() for n, w in modelo.named_parameters() if "weight" in n).sqrt().item()
# >>>
variantes = {"base": entrenar(),
             "dropout 0,5": entrenar(p=0.5),
             "weight decay": entrenar(wd=5e-3)}
for nombre, (modelo, hist) in variantes.items():
    _, lv, av = hist[-1]                 # epoch 200
    print(f"{nombre:12} val {lv:.3f} acc {av:.3f} |W| {norma(modelo):.1f}")
# <<<
json.dump({k: [[e, round(l, 4), round(a, 4)] for e, l, a in h] for k, (_, h) in variantes.items()},
          open(os.path.join(DATOS, "ep08_curvas.json"), "w"))
