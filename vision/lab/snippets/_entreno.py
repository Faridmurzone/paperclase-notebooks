"""Entrenamiento con registro por epoch (train/test) y data augmentation opcional. Lo usan los preparar_ep07/08/09.py."""
import json
import time
import torch
import torch.nn.functional as F


def augmentar(x, pad=4):
    """flip horizontal al azar + recorte aleatorio de 32×32 sobre la imagen con padding de 4 (ceros)."""
    N, C, H, W = x.shape
    voltear = torch.rand(N, device=x.device) < 0.5
    x = torch.where(voltear[:, None, None, None], x.flip(3), x)
    xp = F.pad(x, (pad, pad, pad, pad))
    oy = torch.randint(0, 2 * pad + 1, (N,), device=x.device)
    ox = torch.randint(0, 2 * pad + 1, (N,), device=x.device)
    filas = (oy[:, None] + torch.arange(H, device=x.device))[:, :, None]   # N × H × 1
    cols = (ox[:, None] + torch.arange(W, device=x.device))[:, None, :]    # N × 1 × W
    n = torch.arange(N, device=x.device)[:, None, None]
    return xp.permute(0, 2, 3, 1)[n, filas, cols].permute(0, 3, 1, 2)


@torch.no_grad()
def medir(m, X, y, dev, batch=1000):
    m.eval()
    perd, ok = 0.0, 0
    for i in range(0, len(X), batch):
        out = m(X[i:i + batch].to(dev))
        yb = y[i:i + batch].to(dev)
        perd += F.cross_entropy(out, yb, reduction="sum").item()
        ok += (out.argmax(1) == yb).sum().item()
    return perd / len(X), ok / len(X)


def entrenar(m, Xtr, ytr, Xte, yte, epochs, dev, lr=1e-3, batch=128, aug=False, semilla=0, wd=0.0, nombre="", opt="adam"):
    torch.manual_seed(semilla)
    m.to(dev)
    if opt == "adam":
        o = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    else:
        o = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    pasos = epochs * ((len(Xtr) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(o, max_lr=lr, total_steps=pasos, pct_start=0.15)
    g = torch.Generator().manual_seed(semilla)
    hist = []
    Xtr_d, ytr_d = Xtr.to(dev), ytr.to(dev)
    for ep in range(epochs):
        t0 = time.time()
        m.train()
        orden = torch.randperm(len(Xtr), generator=g).to(dev)
        for i in range(0, len(Xtr), batch):
            idx = orden[i:i + batch]
            xb, yb = Xtr_d[idx], ytr_d[idx]
            if aug:
                xb = augmentar(xb)
            perd = F.cross_entropy(m(xb), yb)
            o.zero_grad()
            perd.backward()
            o.step()
            sched.step()
        ptr, atr = medir(m, Xtr, ytr, dev)
        pte, ate = medir(m, Xte, yte, dev)
        hist.append(dict(epoch=ep + 1, perd_train=round(ptr, 4), acc_train=round(atr, 4), perd_test=round(pte, 4), acc_test=round(ate, 4)))
        print(f"{nombre} epoch {ep + 1}/{epochs}  train {atr:.3f} ({ptr:.3f})  test {ate:.3f}  {time.time() - t0:.0f}s", flush=True)
    m.cpu().eval()
    return hist


def guardar_json(obj, ruta):
    json.dump(obj, open(ruta, "w"), indent=1, ensure_ascii=False)
