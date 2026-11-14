import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _clip import VOCAB, tokens, datos
torch.manual_seed(0)
Xtr, _, _, _ = datos()
# >>>
class EncImagen(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.red = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, d))
    def forward(self, x):                       # x: N × 64 píxeles
        return F.normalize(self.red(x), dim=-1)  # largo 1

class EncTexto(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.emb = nn.EmbeddingBag(len(VOCAB), 64, mode="mean")  # promedio
        self.proy = nn.Linear(64, d)
    def forward(self, frases):
        return F.normalize(self.proy(self.emb(*tokens(frases))), dim=-1)

I, T = EncImagen()(Xtr[:2]), EncTexto()(["un siete", "dígito siete"])
print("vocabulario:", len(VOCAB), "palabras | I:", tuple(I.shape), "T:", tuple(T.shape))
print("largos:", I.norm(dim=-1).tolist(), "| cos(I0, T0) al azar:", round((I[0] @ T[0]).item(), 2))
# <<<
