import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from _transformer import Transformer, BOS, datos   # el resto del transformer (mismo código)
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep10_transformer.pt")
torch.manual_seed(0)
# >>>
@torch.no_grad()
def greedy(modelo, x):
    memoria = modelo.codificar(x)                    # el encoder corre una vez
    y = torch.full((x.shape[0], 1), BOS)
    for _ in range(x.shape[1]):
        logits = modelo.decodificar(y, memoria)[:, -1]   # solo el último
        logits[:, BOS] = float("-inf")               # BOS nunca es salida
        y = torch.cat([y, logits.argmax(-1, keepdim=True)], dim=1)
    return y[:, 1:]                                  # sin el BOS

m = Transformer(); m.load_state_dict(torch.load(CKPT)); m.eval()
x, y, y_in = datos(1000, torch.Generator().manual_seed(123))   # 1000 nuevas
pred = greedy(m, x)
forzado = m(x, y_in).argmax(-1)               # con la respuesta como entrada
for nombre, p in [("greedy", pred), ("teacher forcing", forzado)]:
    print(f"{nombre:15}: secuencias {(p == y).all(1).float().mean():.1%}"
          f" · dígitos {(p == y).float().mean():.1%}")
print("entrada:", *x[0].tolist()); print("salida: ", *pred[0].tolist())
# <<<
