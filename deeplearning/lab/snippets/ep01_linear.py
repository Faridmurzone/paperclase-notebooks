import numpy as np
import torch
from torch import nn
x_np = np.array([1.0, 2.0, 3.0]); w_np = np.array([0.5, -0.5, 0.2]); b_np = 0.5
torch.manual_seed(0)
# >>>
capa = nn.Linear(3, 1)            # 3 entradas, 1 salida
print("weight:", tuple(capa.weight.shape), " bias:", tuple(capa.bias.shape))

with torch.no_grad():             # copiamos nuestros números
    capa.weight.copy_(torch.tensor([[0.5, -0.5, 0.2]]))
    capa.bias.fill_(0.5)

x = torch.tensor([1.0, 2.0, 3.0])
z = capa(x)                       # = x @ weight.T + bias
y = torch.sigmoid(z)
print("torch: z =", round(z.item(), 4), " y =", round(y.item(), 4))

z_np = w_np @ x_np + b_np         # la versión NumPy de antes
print("numpy: z =", round(z_np, 4))
print("iguales:", np.allclose(z.item(), z_np))
# <<<
