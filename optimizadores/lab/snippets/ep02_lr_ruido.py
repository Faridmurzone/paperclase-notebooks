import numpy as np
rng = np.random.default_rng(0)
C = rng.normal(0, 1, size=(1000, 2)); C = C - C.mean(0)
def grad(p, idx): return 2 * (p - C[idx]).mean(0)
def sgd(B, lr=0.1, pasos=40, semilla=1):
    rng = np.random.default_rng(semilla)
    p = np.array([-2.5, 1.5]); camino = [p.copy()]
    for _ in range(pasos):
        p = p - lr * grad(p, rng.choice(1000, B, replace=False))
        camino.append(p.copy())
    return np.array(camino)

# >>>
for B, lr in [(1, 0.1), (1, 0.02), (16, 0.1)]:
    c = sgd(B, lr, pasos=2000)
    cola = np.linalg.norm(c[1000:], axis=1)    # ya llegó: medimos el temblor
    print(f"B = {B:2}, lr = {lr:4}: distancia media al mínimo {cola.mean():.3f}")
# <<<
