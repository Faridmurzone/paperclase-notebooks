import numpy as np
import torch
from torch import nn
X_np = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [2.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
# >>>
capa = nn.Linear(3, 2)                 # 3 entradas, 2 neuronas
print("weight:", tuple(capa.weight.shape), " bias:", tuple(capa.bias.shape))

with torch.no_grad():
    capa.weight.copy_(torch.tensor([[0.5, -0.5, 0.2],
                                    [-0.3, 0.1, 0.4]]))
    capa.bias.copy_(torch.tensor([0.5, 0.0]))

X = torch.tensor(X_np, dtype=torch.float32)  # el batch de 4
Z = capa(X)                            # X @ weight.T + bias
print("Z:", tuple(Z.shape))
print("igual a NumPy:", np.allclose(Z.detach().numpy(),
                                    X_np @ capa.weight.detach().numpy().T + [0.5, 0.0]))
print("parámetros:", sum(p.numel() for p in capa.parameters()))
# <<<
