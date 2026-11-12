import numpy as np
# >>>
ang = np.radians([90, 210, 330])           # 3 direcciones a 120°
W = np.stack([np.cos(ang), np.sin(ang)])   # 2 × 3: 3 features en 2 dims
print("W =\n", W.round(2))
print("WᵀW =\n", (W.T @ W).round(2))       # cosenos entre features

f = np.array([1.0, 0.0, 0.0])              # solo la feature 0 activa
x = W @ f                                  # comprimir a 2 números
leido = W.T @ x                            # leer las 3 con productos punto
print("x =", x.round(2), "  leído =", leido.round(2))
print("ReLU(leído) =", np.maximum(0, leido).round(2))

f2 = np.array([1.0, 1.0, 0.0])             # dos activas a la vez
print("con dos activas:", np.maximum(0, W.T @ W @ f2).round(2))
# <<<
