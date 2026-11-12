import numpy as np
# >>>
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print("A.T =\n", A.T)                  # filas ↔ columnas
print("A[0, 2] =", A[0, 2], " A.T[2, 0] =", A.T[2, 0])

x = np.array([[1], [2], [3]])          # vectores columna: 3 × 1
y = np.array([[4], [0], [1]])
print("xᵀ y =", (x.T @ y).item(), " (1×3 por 3×1)")
print("x yᵀ forma:", (x @ y.T).shape, " (3×1 por 1×3)")

B = np.array([[1, 0], [0, 1], [1, 1]])
print("(A B)ᵀ == Bᵀ Aᵀ:", np.array_equal((A @ B).T, B.T @ A.T))
# <<<
