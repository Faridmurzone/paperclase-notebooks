import json, os
import numpy as np
import torch
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
def grad(p): return np.array([0.2, 6.0]) * p
def valle(w): return 0.1 * w[0] ** 2 + 3 * w[1] ** 2
camino = [[-4.5, 1.5]]

# >>>
mu, lr = 0.9, 0.03
th, v = np.array([-4.5, 1.5]), np.zeros(2)       # Sutskever et al., 2013
w = torch.tensor([-4.5, 1.5], requires_grad=True)
opt = torch.optim.SGD([w], lr=lr, momentum=mu, nesterov=True)
for t in range(40):
    v = mu * v - lr * grad(th + mu * v)          # gradiente adelantado
    th = th + v
    opt.zero_grad(); valle(w).backward(); opt.step()
    camino.append(th.tolist())
adelantado = th + mu * v
print("Sutskever, θ + μv:", adelantado.round(4))
print("torch, nesterov:  ", w.detach().numpy().round(4))
print(f"distancia al mínimo: {np.linalg.norm(th):.3f}")
# <<<
json.dump(np.round(camino, 4).tolist(), open(os.path.join(DATOS, "ep03_nesterov.json"), "w"))
