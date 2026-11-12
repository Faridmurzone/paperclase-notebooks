import numpy as np
rng = np.random.default_rng(0)
# >>>
rango = np.linalg.matrix_rank
print("identidad 3×3     :", rango(np.eye(3)))
print("[[1,2],[2,4]]     :", rango(np.array([[1, 2], [2, 4]])))

u1, v1 = rng.normal(size=6), rng.normal(size=6)
u2, v2 = rng.normal(size=6), rng.normal(size=6)
print("u1 v1ᵀ            :", rango(np.outer(u1, v1)))
print("u1 v1ᵀ + u2 v2ᵀ   :", rango(np.outer(u1, v1) + np.outer(u2, v2)))

B = rng.normal(size=(6, 2))          # 6 × 2
A = rng.normal(size=(2, 6))          # 2 × 6
print("B @ A (6×6)       :", rango(B @ A))
print("al azar 6×6       :", rango(rng.normal(size=(6, 6))))
# <<<
