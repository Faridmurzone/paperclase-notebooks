import torch
# >>>
x = torch.tensor([1.0, 2.0, 3.0])          # la entrada: 3 números
W0 = torch.eye(3)                          # pesos congelados: identidad
A = torch.tensor([[1.0, 0.0, 1.0]])        # r × k = 1 × 3
B = torch.tensor([[0.5], [0.0], [-0.5]])   # d × r = 3 × 1
alfa, r = 1.0, 1
s = alfa / r                               # la escala

Ax = A @ x                                 # r números: lo que lee
cambio = s * B @ Ax                        # d números: lo que escribe
h = W0 @ x + cambio
print("A x      =", Ax.tolist())
print("s B (Ax) =", cambio.tolist())
print("h        =", h.tolist())
print("igual con (W0 + s BA) x:", torch.equal(h, (W0 + s * B @ A) @ x))
# <<<
