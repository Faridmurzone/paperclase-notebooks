import numpy as np
W = np.array([[0.5, -0.5, 0.2], [-0.3, 0.1, 0.4]]); b = np.array([0.5, 0.0])
# >>>
X = np.array([[1.0, 2.0, 3.0],     # ejemplo 0: el de antes
              [0.0, 1.0, 0.0],
              [2.0, 0.0, 1.0],
              [1.0, 1.0, 1.0]])    # batch de 4 ejemplos

Z = X @ W.T + b                    # (4,3) @ (3,2) + (2,)
print("X:", X.shape, " W.T:", W.T.shape, " Z:", Z.shape)
print(Z.round(2))
print("ReLU fila 2:", np.maximum(0, Z[2]))
print("fila 0 == z de antes:", np.allclose(Z[0], W @ X[0] + b))
# <<<
