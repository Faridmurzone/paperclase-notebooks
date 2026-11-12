import torch
from torch import nn
from sklearn.datasets import load_digits
torch.manual_seed(0)
# >>>
X = torch.tensor(load_digits().data[:16], dtype=torch.float32)  # 16 dígitos 8x8

red = nn.Sequential(nn.Linear(64, 32),   # 64 píxeles -> 32 neuronas
                    nn.ReLU(),
                    nn.Linear(32, 10))   # 32 -> 10 (un número por dígito)

h = X
for capa in red:
    h = capa(h)
    n = sum(p.numel() for p in capa.parameters())
    print(f"{capa.__class__.__name__:7} -> {tuple(h.shape)}  parámetros: {n}")
# <<<
