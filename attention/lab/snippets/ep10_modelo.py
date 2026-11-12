import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from _transformer import V, pe, CapaEncoder, CapaDecoder   # el resto del transformer (mismo código)
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep10_transformer.pt")
torch.manual_seed(0)
# >>>
class Transformer(nn.Module):
    def __init__(s, d=32, h=4, N=2, d_ff=128, p=0.1):
        super().__init__()
        s.d, s.emb = d, nn.Embedding(V, d)           # una sola matriz
        s.register_buffer("pe", pe(64, d))           # senos del episodio 4
        s.enc = nn.ModuleList(CapaEncoder(d, h, d_ff, p) for _ in range(N))
        s.dec = nn.ModuleList(CapaDecoder(d, h, d_ff, p) for _ in range(N))
        s.drop = nn.Dropout(p)
    def entrada(s, t):                               # embedding · √d + posición
        return s.drop(s.emb(t) * math.sqrt(s.d) + s.pe[:t.shape[1]])
    def codificar(s, x):
        m = s.entrada(x)
        for capa in s.enc: m = capa(m)
        return m                                     # la memoria
    def decodificar(s, y_in, memoria):
        T = y_in.shape[1]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        y = s.entrada(y_in)
        for capa in s.dec: y = capa(y, memoria, causal)
        return y @ s.emb.weight.T                    # logits: la misma matriz
    def forward(s, x, y_in):
        return s.decodificar(y_in, s.codificar(x))

m = Transformer()
print(f"parámetros: {sum(p.numel() for p in m.parameters()):,}"
      f" · embeddings: {m.emb.weight.numel()} (una sola vez)")
# <<<
