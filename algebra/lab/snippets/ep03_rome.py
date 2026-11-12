import numpy as np
# >>>
W = np.array([[1., 2., 0.],
              [0., 1., 1.],
              [1., 0., 1.]])
k = np.array([1., 1., 0.])           # la clave: «el sujeto»
v = np.array([0., 0., 5.])           # el valor nuevo que queremos
print("W k antes:", W @ k)

delta = np.outer(v - W @ k, k) / (k @ k)    # rango uno
W2 = W + delta
print("W' k     :", W2 @ k)

x = np.array([1., -1., 0.])          # perpendicular a k
print("x ⟂ k:  W x =", W @ x, "  W' x =", W2 @ x)
print("rango(delta) =", np.linalg.matrix_rank(delta))
# <<<
