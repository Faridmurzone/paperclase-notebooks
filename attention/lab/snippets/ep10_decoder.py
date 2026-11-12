import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from _transformer import MultiHead, ffn   # el resto del transformer (mismo código)
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep10_transformer.pt")
torch.manual_seed(0)
# >>>
class CapaDecoder(nn.Module):
    def __init__(s, d, h, d_ff, p):
        super().__init__()
        s.att, s.cruzada, s.ffn = MultiHead(d, h), MultiHead(d, h), ffn(d, d_ff)
        s.ln = nn.ModuleList(nn.LayerNorm(d) for _ in range(3))
        s.drop = nn.Dropout(p)

    def forward(s, y, memoria, causal):
        y = s.ln[0](y + s.drop(s.att(y, y, causal)))          # self, con máscara
        y = s.ln[1](y + s.drop(s.cruzada(y, memoria)))        # cross-attention
        return s.ln[2](y + s.drop(s.ffn(y)))

capa = CapaDecoder(d=32, h=4, d_ff=128, p=0.1).eval()
memoria, y = torch.randn(2, 10, 32), torch.randn(2, 7, 32)  # 10 de entrada, 7 escritos
causal = torch.triu(torch.ones(7, 7, dtype=torch.bool), diagonal=1)
print("sale", tuple(capa(y, memoria, causal).shape),
      "· parámetros", sum(p.numel() for p in capa.parameters()))
# <<<
