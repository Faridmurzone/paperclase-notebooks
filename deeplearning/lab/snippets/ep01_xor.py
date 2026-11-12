import numpy as np
# >>>
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0])       # XOR: 1 si son distintas

w, b, eta = np.zeros(2), 0.0, 0.1
for epoch in range(1, 101):
    errores = 0
    for xi, yi in zip(X, y):
        error = yi - float(w @ xi + b > 0)
        w += eta * error * xi
        b += eta * error
        errores += int(error != 0)
    if epoch in (1, 2, 3, 10, 50, 100):
        print(f"epoch {epoch:3}: errores={errores}")
print("predice:", (X @ w + b > 0).astype(int), " quería:", y)
# <<<
