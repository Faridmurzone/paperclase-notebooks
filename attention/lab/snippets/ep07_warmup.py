import math
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(1)


def datos(n, T=10, V=10):
    """La tarea del episodio 6: para cada token, cuántas veces MÁS aparece (0..9)."""
    x = torch.randint(0, V, (n, T))
    return x, (x[:, :, None] == x[:, None, :]).sum(-1) - 1


def pila(N=12, d=32):
    """La pila post-norm de 12 capas del episodio 6 (misma inicialización)."""
    torch.manual_seed(0)
    capa = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True)
    return nn.Sequential(nn.Embedding(10, d), nn.TransformerEncoder(capa, N, enable_nested_tensor=False), nn.Linear(d, 10))


# >>>
forma = lambda s, w: min(s / w, math.sqrt(w / s))   # la de la ecuación (3)

def entrenar(warmup, pico=1e-3, pasos=400):         # post-norm, 12 capas
    m = pila()
    opt = torch.optim.Adam(m.parameters(), lr=pico, betas=(0.9, 0.98), eps=1e-9)
    f = (lambda s: forma(s + 1, warmup)) if warmup else (lambda s: 1.0)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, f)
    torch.manual_seed(1); perdidas = []
    for paso in range(1, pasos + 1):
        x, y = datos(64)
        loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if paso % 100 == 0: perdidas.append(f"{loss.item():.2f}")
    x, y = datos(2000)
    return perdidas, (m(x).argmax(-1) == y).float().mean().item()

for w in (0, 100):
    perdidas, acc = entrenar(w)
    print(f"warmup {w:3}: pérdida c/100 pasos", *perdidas, f"· {acc:.0%}")
# <<<
