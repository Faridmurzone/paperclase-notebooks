import json, os
import numpy as np
AQUI = os.path.dirname(os.path.abspath(__file__))
# >>>
rng = np.random.default_rng(1)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0], dtype=float)
W1, b1 = rng.normal(0, 1, (4, 2)), np.zeros(4)   # 4 neuronas ocultas
w2, b2 = rng.normal(0, 1, 4), 0.0
eta, N = 0.1, len(X)

for paso in range(501):
    Z1 = X @ W1.T + b1;  H = np.maximum(0, Z1)        # forward
    y_hat = H @ w2 + b2;  L = ((y_hat - y) ** 2).mean()
    d = 2 * (y_hat - y) / N                           # backward
    dw2, db2 = H.T @ d, d.sum()
    dZ1 = np.outer(d, w2) * (Z1 > 0)
    dW1, db1 = dZ1.T @ X, dZ1.sum(0)
    W1 -= eta * dW1;  b1 -= eta * db1                 # un paso
    w2 -= eta * dw2;  b2 -= eta * db2
    if paso in (0, 100, 500):
        print(f"paso {paso:3}: pérdida {L:.4f}")
print("predice:", y_hat.round(2))
# <<<
# curva de pérdida para el video (mismo cálculo, reiniciando)
rng = np.random.default_rng(1)
W1, b1 = rng.normal(0, 1, (4, 2)), np.zeros(4); w2, b2 = rng.normal(0, 1, 4), 0.0
curva = []
for paso in range(501):
    Z1 = X @ W1.T + b1; H = np.maximum(0, Z1); y_hat = H @ w2 + b2; L = ((y_hat - y) ** 2).mean()
    if paso % 10 == 0: curva.append([paso, round(float(L), 4)])
    d = 2 * (y_hat - y) / N; dw2, db2 = H.T @ d, d.sum(); dZ1 = np.outer(d, w2) * (Z1 > 0); dW1, db1 = dZ1.T @ X, dZ1.sum(0)
    W1 -= eta * dW1; b1 -= eta * db1; w2 -= eta * dw2; b2 -= eta * db2
json.dump(curva, open(os.path.join(AQUI, "..", "datos", "ep05_xor_curva.json"), "w"))
