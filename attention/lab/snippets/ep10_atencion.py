import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from _transformer import V, BOS, datos   # el resto del transformer (mismo código)
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep10_transformer.pt")
torch.manual_seed(0)
# >>>
class MultiHead(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.h, s.dk = h, d // h
        s.wq, s.wk, s.wv, s.wo = (nn.Linear(d, d) for _ in range(4))

    def forward(s, xq, xkv, prohibido=None):       # prohibido: True = no mirar
        B, Tq, Tk = xq.shape[0], xq.shape[1], xkv.shape[1]
        q = s.wq(xq).view(B, Tq, s.h, s.dk).transpose(1, 2)    # [B, h, Tq, dk]
        k = s.wk(xkv).view(B, Tk, s.h, s.dk).transpose(1, 2)
        v = s.wv(xkv).view(B, Tk, s.h, s.dk).transpose(1, 2)
        S = q @ k.transpose(-2, -1) / s.dk ** 0.5               # [B, h, Tq, Tk]
        if prohibido is not None:
            S = S.masked_fill(prohibido, float("-inf"))
        out = S.softmax(-1) @ v                                 # [B, h, Tq, dk]
        return s.wo(out.transpose(1, 2).reshape(B, Tq, -1))

att, x = MultiHead(d=32, h=4), torch.randn(2, 5, 32)        # batch 2, 5 tokens
causal = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
x2 = x.clone(); x2[:, 4] = torch.randn(32)                  # cambio el último
print("salida", tuple(att(x, x, causal).shape), "· tokens 0–3 iguales:",
      torch.allclose(att(x, x, causal)[:, :4], att(x2, x2, causal)[:, :4]))
# <<<
