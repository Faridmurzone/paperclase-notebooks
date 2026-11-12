import json, os
import numpy as np
import torch
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
def grad(p): return np.array([0.2, 6.0]) * p      # valle: 0,1x² + 3y²

# >>>
class Momentum:
    def __init__(self, lr, mu):
        self.lr, self.mu, self.v = lr, mu, 0.0
    def paso(self, p, g):
        self.v = self.mu * self.v + g             # recordar los gradientes
        return p - self.lr * self.v               # moverse con la suma

opt_mano = Momentum(lr=0.03, mu=0.9)
p = np.array([-4.5, 1.5]); camino = [p]
w = torch.tensor([-4.5, 1.5], requires_grad=True)
opt = torch.optim.SGD([w], lr=0.03, momentum=0.9)
for t in range(40):
    p = opt_mano.paso(p, grad(p)); camino.append(p)
    opt.zero_grad(); (0.1 * w[0]**2 + 3 * w[1]**2).backward(); opt.step()
print("mano: ", p.round(4), " torch:", w.detach().numpy().round(4))
print(f"distancia al mínimo: {np.linalg.norm(p):.3f}")
# <<<
json.dump(np.round(camino, 4).tolist(), open(os.path.join(DATOS, "ep03_momentum.json"), "w"))
