import numpy as np
np.set_printoptions(precision=2, suppress=True)
# >>>
A = np.array([[2.0, 1.0],
              [1.0, 2.0]])            # una matriz 2 × 2

x = np.array([1.0, 0.0])
print("A @ (1, 0) =", A @ x)          # cambia de dirección

u = np.array([1.0, 1.0])
print("A @ (1, 1) =", A @ u)          # misma dirección, ×3

lam, Q = np.linalg.eigh(A)            # autovalores y autovectores
print("autovalores:", lam)
print("autovectores (columnas):")
print(Q)
print("A @ q2 - 3 * q2 =", A @ Q[:, 1] - lam[1] * Q[:, 1])
# <<<
