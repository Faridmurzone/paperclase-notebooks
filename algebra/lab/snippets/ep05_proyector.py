import numpy as np
import torch
torch.manual_seed(0)
# >>>
r_hat = np.array([0.6, 0.8])
x = np.array([2.0, 1.0])

P = np.outer(r_hat, r_hat)             # r̂ r̂ᵀ, el producto exterior
print("P =\n", P.round(2))
print("P @ x       =", (P @ x).round(2))           # la sombra
print("(I - P) @ x =", ((np.eye(2) - P) @ x).round(2))
print("P @ P == P:", np.allclose(P @ P, P))

d = 2304                               # el tamaño de Gemma
r = torch.randn(d); r = r / r.norm()
Q = torch.eye(d) - torch.outer(r, r)
print("rango de I - r̂r̂ᵀ con d = 2304:", torch.linalg.matrix_rank(Q).item())
# <<<
