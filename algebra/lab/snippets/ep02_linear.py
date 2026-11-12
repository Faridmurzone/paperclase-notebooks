import torch
from torch import nn
torch.manual_seed(0)
# >>>
lin = nn.Linear(4, 3)            # entra d=4, sale 3
print("weight:", tuple(lin.weight.shape), " (salida, entrada)")

X = torch.randn(5, 4)            # 5 tokens, uno por fila
Y = lin(X)
print("X:", tuple(X.shape), "→ Y:", tuple(Y.shape))

a_mano = X @ lin.weight.T + lin.bias
print("igual a X @ W.T + b:", torch.allclose(Y, a_mano))

fila = lin(X[2])                  # un solo token, aparte
print("fila 2 suelta = fila 2 del lote:", torch.allclose(fila, Y[2]))
# <<<
