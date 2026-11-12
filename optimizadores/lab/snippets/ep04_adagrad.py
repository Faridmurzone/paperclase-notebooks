import json, os
import numpy as np
import torch
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
def grad(p): return np.array([0.2, 6.0]) * p          # valle: 0,1x² + 3y²
def valle(w): return 0.1 * w[0] ** 2 + 3 * w[1] ** 2

# >>>
class AdaGrad:
    def __init__(self, lr, eps=1e-10):
        self.lr, self.eps, self.s = lr, eps, 0.0
    def paso(self, p, g):
        self.s = self.s + g ** 2                     # suma de cuadrados
        return p - self.lr * g / (np.sqrt(self.s) + self.eps)

mano, p = AdaGrad(lr=0.5), np.array([-4.5, 1.5]); camino = [p]
w = torch.tensor([-4.5, 1.5], requires_grad=True)
opt = torch.optim.Adagrad([w], lr=0.5)
for t in range(40):
    p = mano.paso(p, grad(p)); camino.append(p)
    opt.zero_grad(); valle(w).backward(); opt.step()
print("primer paso:", (camino[1] - camino[0]).round(3))
print("mano: ", p.round(4), " torch:", w.detach().numpy().round(4))
print(f"diferencia máxima: {np.abs(p - w.detach().numpy()).max():.1e}")
# <<<
json.dump(np.round(camino, 4).tolist(), open(os.path.join(DATOS, "ep04_adagrad.json"), "w"))
