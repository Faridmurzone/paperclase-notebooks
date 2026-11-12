import numpy as np
# >>>
u = np.array([1, 2, 3])
v = np.array([1, 0, -1])

M = np.outer(u, v)                 # M[i, j] = u[i] * v[j]
print("u vᵀ =\n", M)

x = np.array([2, 1, 1])
print("M @ x      =", M @ x)
print("v·x        =", v @ x)        # cuánto de v tiene x
print("u * (v·x)  =", u * (v @ x))  # escribe u, con ese peso
print("rango(M)   =", np.linalg.matrix_rank(M))
# <<<
