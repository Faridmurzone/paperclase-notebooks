import numpy as np
class AdaGrad:
    def __init__(self, lr, eps=1e-10):
        self.lr, self.eps, self.s = lr, eps, 0.0
    def paso(self, p, g):
        self.s = self.s + g ** 2
        return p - self.lr * g / (np.sqrt(self.s) + self.eps)

# >>>
opt, w = AdaGrad(lr=1.0), 0.0
for t in range(1, 1001):
    nuevo = opt.paso(w, 0.5)            # pendiente constante: g = 0,5
    if t in (1, 4, 100, 1000):
        print(f"paso {t:4}: se movió {abs(nuevo - w):.4f}"
              f"   recorrido total {abs(nuevo):7.2f}")
    w = nuevo
# <<<
