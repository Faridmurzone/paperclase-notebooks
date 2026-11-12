import numpy as np
# >>>
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0], dtype=float)          # XOR

# el MEJOR modelo lineal posible (mínimos cuadrados): y ≈ X w + b
A = np.hstack([X, np.ones((4, 1))])              # columna de unos = sesgo
w1, w2, b = np.linalg.lstsq(A, y, rcond=None)[0]
print("w =", np.round([w1, w2], 3) + 0.0, " b =", round(b, 3))
print("predice:", (A @ [w1, w2, b]).round(3))
# <<<
