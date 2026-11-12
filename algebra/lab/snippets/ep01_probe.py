import numpy as np
# >>>
rng = np.random.default_rng(0)
pos = rng.normal([2.0, 1.0], 0.5, (20, 2))    # 20 puntos «sí»
neg = rng.normal([-1.0, 1.5], 0.5, (20, 2))   # 20 puntos «no»

r = pos.mean(axis=0) - neg.mean(axis=0)
print("r =", r.round(2))

X = np.vstack([pos, neg])
y = np.array([1] * 20 + [0] * 20)
puntaje = X @ r              # un producto punto por fila
tau = (pos.mean(0) @ r + neg.mean(0) @ r) / 2   # a mitad

y_hat = (puntaje > tau).astype(int)
print("puntajes (3 sí):", puntaje[:3].round(2))
print("puntajes (3 no):", puntaje[20:23].round(2))
print("tau =", round(tau, 2), " aciertos:", (y_hat == y).mean())
# <<<
