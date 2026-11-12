import numpy as np
np.set_printoptions(precision=2, suppress=True)
# >>>
M = np.array([[3.0, 1.0, 1.0],
              [-1.0, 3.0, 1.0]])      # 2 × 3: no es cuadrada

U, S, Vt = np.linalg.svd(M, full_matrices=False)
print("U:", U.shape, " S:", S.shape, " Vt:", Vt.shape)
print("valores singulares:", S)
print("reconstruye M:", np.allclose(U @ np.diag(S) @ Vt, M))

M1 = S[0] * np.outer(U[:, 0], Vt[0])  # solo el término 1
error = np.linalg.norm(M - M1)        # norma de Frobenius
print("rango de M1:", np.linalg.matrix_rank(M1))
print(f"error de M1: {error:.2f}   sigma_2 = {S[1]:.2f}")
# <<<
