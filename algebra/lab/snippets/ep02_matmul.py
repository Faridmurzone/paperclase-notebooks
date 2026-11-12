import numpy as np
# >>>
A = np.array([[1, 2, 3], [4, 5, 6]])     # 2 × 3
B = np.array([[1, 0], [0, 1], [1, 1]])   # 3 × 2

C = np.zeros((2, 2), dtype=int)
for i in range(2):                       # fila i de A
    for j in range(2):                   # columna j de B
        C[i, j] = A[i] @ B[:, j]
print("a mano:\n", C)
print("A @ B :\n", A @ B)
print("(B @ A).shape =", (B @ A).shape)  # otro orden, otra cosa

try:
    A @ A                                # 2×3 por 2×3
except ValueError as e:
    print("A @ A: error, 3 ≠ 2")
# <<<
