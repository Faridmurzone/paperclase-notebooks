import json, os
import numpy as np
import torch
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
def grad(p): return np.array([0.2, 6.0]) * p          # valle: 0,1x² + 3y²
def valle(w): return 0.1 * w[0] ** 2 + 3 * w[1] ** 2

# >>>
class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = 0.0, 0.0, 0
    def paso(self, p, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g        # 1er momento
        self.v = self.b2 * self.v + (1 - self.b2) * g ** 2   # 2do momento
        m_hat = self.m / (1 - self.b1 ** self.t)             # corrección
        v_hat = self.v / (1 - self.b2 ** self.t)
        return p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

mano, p = Adam(lr=0.1), np.array([-4.5, 1.5]); camino = [p]
w = torch.tensor([-4.5, 1.5], requires_grad=True)
opt = torch.optim.Adam([w], lr=0.1)
for t in range(40):
    p = mano.paso(p, grad(p)); camino.append(p)
    opt.zero_grad(); valle(w).backward(); opt.step()
print("mano: ", p.round(4), " torch:", w.detach().numpy().round(4))
print(f"diferencia máxima: {np.abs(p - w.detach().numpy()).max():.1e}")
# <<<
json.dump(np.round(camino, 4).tolist(), open(os.path.join(DATOS, "ep05_adam.json"), "w"))
