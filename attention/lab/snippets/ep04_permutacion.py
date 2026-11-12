import math
import torch
import torch.nn as nn
torch.manual_seed(0)
d = 16
emb = nn.Embedding(10, d)
attn = nn.MultiheadAttention(d, 2, batch_first=True)
pe = torch.zeros(5, d)
p = torch.arange(5).unsqueeze(1); om = 1 / 10000 ** (torch.arange(0, d, 2) / d)
pe[:, 0::2], pe[:, 1::2] = torch.sin(p * om), torch.cos(p * om)
vocab = {"el": 0, "perro": 1, "mordió": 2, "al": 3, "hombre": 4}
def ids(frase):
    return torch.tensor([[vocab[w] for w in frase.split()]])
def salida(frase, con_pe):
    x = emb(ids(frase)) + (pe if con_pe else 0)
    return attn(x, x, x, need_weights=False)[0][0]
# >>>
a = "el perro mordió al hombre"
b = "el hombre mordió al perro"
for con_pe in (False, True):
    sa, sb = salida(a, con_pe), salida(b, con_pe)
    igual = torch.allclose(sa[2], sb[2], atol=1e-6)  # «mordió», pos 2
    perm = torch.allclose(sa[[0, 4, 2, 3, 1]], sb, atol=1e-6)
    print(f"PE={con_pe!s:5}  «mordió» igual en las dos: {igual!s:5}"
          f"  b = a permutada: {perm}")
# <<<
