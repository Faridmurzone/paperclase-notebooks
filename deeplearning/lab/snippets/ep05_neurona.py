import torch
# >>>
x, w, b, y = 2.0, 3.0, 1.0, 5.0

# forward: guardamos cada valor intermedio
z = w * x + b               # 7
L = (z - y) ** 2            # 4

# backward a mano: regla de la cadena, de atrás hacia adelante
dL_dz = 2 * (z - y)         # dL/dz
dL_dw = dL_dz * x           # dz/dw = x
dL_db = dL_dz * 1           # dz/db = 1
dL_dx = dL_dz * w           # dz/dx = w
print(f"z = {z}  L = {L}")
print(f"dL/dz = {dL_dz}  dL/dw = {dL_dw}")
print(f"dL/db = {dL_db}  dL/dx = {dL_dx}")

# lo mismo con autograd de PyTorch (lo abrimos en el ep. 6)
wt = torch.tensor(w, requires_grad=True); bt = torch.tensor(b, requires_grad=True)
((wt * x + bt - y) ** 2).backward()
print("PyTorch: dL/dw =", wt.grad.item(), " dL/db =", bt.grad.item())
# <<<
