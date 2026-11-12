import numpy as np

# >>>
def correr(regla, k, lr, pasos=40):     # pérdida: k · (0,1x² + 3y²)
    p, s = np.array([-4.5, 1.5]), 0.0
    for _ in range(pasos):
        g = k * np.array([0.2, 6.0]) * p
        if regla == "SGD":
            p = p - lr * g
        else:                           # AdaGrad
            s = s + g ** 2
            p = p - lr * g / (np.sqrt(s) + 1e-10)
    return p

for regla, lr in [("SGD", 0.3), ("AdaGrad", 0.5)]:
    for k in [1, 100]:
        p = correr(regla, k, lr)
        fin = "diverge" if np.abs(p).max() > 1e6 else p.round(3)
        print(f"{regla:7} k = {k:3}: termina en {fin}")
# <<<
