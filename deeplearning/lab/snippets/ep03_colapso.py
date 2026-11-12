import torch
from torch import nn
torch.manual_seed(0)
# >>>
capa1 = nn.Linear(3, 4)            # 3 -> 4, SIN activación
capa2 = nn.Linear(4, 2)            # 4 -> 2

# una sola capa equivalente: W = W2 W1,  b = W2 b1 + b2
W = capa2.weight @ capa1.weight                  # (2,4)@(4,3) = (2,3)
b = capa2.weight @ capa1.bias + capa2.bias

X = torch.randn(5, 3)              # un batch de 5
dos_capas = capa2(capa1(X))
una_capa = X @ W.T + b
print("W:", tuple(W.shape), " b:", tuple(b.shape))
print("iguales:", torch.allclose(dos_capas, una_capa, atol=1e-6))
# <<<
