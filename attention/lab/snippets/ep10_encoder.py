import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from _transformer import MultiHead   # el resto del transformer (mismo código)
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep10_transformer.pt")
torch.manual_seed(0)
# >>>
def ffn(d, d_ff):
    return nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(), nn.Linear(d_ff, d))

class CapaEncoder(nn.Module):
    def __init__(s, d, h, d_ff, p):
        super().__init__()
        s.att, s.ffn = MultiHead(d, h), ffn(d, d_ff)
        s.ln = nn.ModuleList(nn.LayerNorm(d) for _ in range(2))
        s.drop = nn.Dropout(p)

    def forward(s, x):                                # post-norm, como el paper
        x = s.ln[0](x + s.drop(s.att(x, x)))
        return s.ln[1](x + s.drop(s.ffn(x)))

capa = CapaEncoder(d=32, h=4, d_ff=128, p=0.1).eval()
x = torch.randn(2, 10, 32)
print("entra", tuple(x.shape), "sale", tuple(capa(x).shape),
      "· parámetros", sum(p.numel() for p in capa.parameters()))
# <<<
