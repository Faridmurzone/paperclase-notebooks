import json, os
import numpy as np
import torch
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
def grad(p): return np.array([0.2, 6.0]) * p
def valle(w): return 0.1 * w[0] ** 2 + 3 * w[1] ** 2

# >>>
class RMSProp:
    def __init__(self, lr, alpha=0.9, eps=1e-8):
        self.lr, self.alpha, self.eps, self.v = lr, alpha, eps, 0.0
    def paso(self, p, g):
        self.v = self.alpha * self.v + (1 - self.alpha) * g ** 2
        return p - self.lr * g / (np.sqrt(self.v) + self.eps)

mano, p = RMSProp(lr=0.1), np.array([-4.5, 1.5]); camino = [p]
w = torch.tensor([-4.5, 1.5], requires_grad=True)
opt = torch.optim.RMSprop([w], lr=0.1, alpha=0.9, eps=1e-8)
for t in range(40):
    p = mano.paso(p, grad(p)); camino.append(p)
    opt.zero_grad(); valle(w).backward(); opt.step()
print("primer paso:", (camino[1] - camino[0]).round(3))
print("mano: ", p.round(4), " torch:", w.detach().numpy().round(4))
print(f"diferencia máxima: {np.abs(p - w.detach().numpy()).max():.1e}")
# <<<
class AdaGrad:
    def __init__(self, lr):
        self.lr, self.s = lr, 0.0
    def paso(self, p, g):
        self.s = self.s + g ** 2
        return p - self.lr * g / (np.sqrt(self.s) + 1e-10)
a, q, ca = AdaGrad(0.1), np.array([-4.5, 1.5]), [[-4.5, 1.5]]
for t in range(40):
    q = a.paso(q, grad(q)); ca.append(q)
json.dump({"rmsprop": np.round(camino, 4).tolist(), "adagrad01": np.round(ca, 4).tolist()},
          open(os.path.join(DATOS, "ep04_rmsprop.json"), "w"))
