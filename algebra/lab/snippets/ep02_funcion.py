import numpy as np
# >>>
A = np.array([[2, -1],
              [1,  1]])
e1, e2 = np.array([1, 0]), np.array([0, 1])
print("A @ e1 =", A @ e1, " (columna 0)")
print("A @ e2 =", A @ e2, " (columna 1)")

x = np.array([1, 2])                 # x = 1·e1 + 2·e2
print("A @ x  =", A @ x, " = 1·(A e1) + 2·(A e2)")

R = np.array([[0, -1], [1, 0]])      # rotar 90°
S = np.array([[2, 0], [0, 0.5]])     # estirar x, achicar y
print("R @ x  =", R @ x)
print("S @ x  =", S @ x)
print("R @ (S @ x) =", R @ (S @ x), " = (R @ S) @ x =", (R @ S) @ x)
# <<<
