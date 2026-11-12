import torch
# >>>
x, t = 2.0, 1.0                          # entrada y objetivo
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(-0.5, requires_grad=True)

z = w * x + b                            # forward, paso a paso
y = torch.sigmoid(z)
L = (y - t) ** 2
print(f"z={z.item():.2f}  y={y.item():.4f}  L={L.item():.4f}")

dL_dy = 2 * (y - t)                      # derivadas locales
dy_dz = y * (1 - y)
dz_dw = x
print(f"locales: {dL_dy.item():.4f} {dy_dz.item():.4f} {dz_dw}")
print(f"a mano: dL/dw = {(dL_dy * dy_dz * dz_dw).item():.4f}")

L.backward()                             # autograd
print(f"autograd: dL/dw = {w.grad.item():.4f}  dL/db = {b.grad.item():.4f}")
# <<<
