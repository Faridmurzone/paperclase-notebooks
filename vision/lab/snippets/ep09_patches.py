import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _cifar import cargar, normalizar
X, _ = cargar("test", n=1)
x = normalizar(X)                            # (1, 3, 32, 32)
torch.manual_seed(0)
# >>>
p, d = 4, 128
patches = x.unfold(2, p, p).unfold(3, p, p)          # (1, 3, 8, 8, 4, 4)
patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(1, 64, 3 * p * p)
print("patches:", tuple(patches.shape))              # 64 tokens de 48

proy = torch.nn.Conv2d(3, d, kernel_size=p, stride=p)   # patch embedding
tok_conv = proy(x).flatten(2).transpose(1, 2)        # (1, 64, 128)
W = proy.weight.reshape(d, -1)                       # (128, 48)
tok_lineal = patches @ W.T + proy.bias               # una capa lineal
print("tokens:", tuple(tok_conv.shape),
      " dif. máx:", f"{(tok_conv - tok_lineal).abs().max().item():.1e}")
# <<<
