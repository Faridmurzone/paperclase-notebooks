import numpy as np
# >>>
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 0, 0, 1])       # AND: 1 solo si las dos son 1

w, b, eta = np.zeros(2), 0.0, 0.1   # eta: learning rate
for epoch in range(1, 11):
    errores = 0
    for xi, yi in zip(X, y):
        y_hat = float(w @ xi + b > 0)     # escalón
        error = yi - y_hat                # +1, -1 o 0
        w += eta * error * xi             # corregir hacia xi
        b += eta * error
        errores += int(error != 0)
    print(f"epoch {epoch}: errores={errores}  w={w.round(2)} b={b:.1f}")
    if errores == 0:
        break
print("predice:", (X @ w + b > 0).astype(int))
# <<<
