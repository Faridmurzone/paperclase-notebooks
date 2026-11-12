import numpy as np
# >>>
A = np.array([[1, 2, 3],
              [4, 5, 6]])         # 2 filas, 3 columnas

print("forma    :", A.shape)       # (filas, columnas)
print("fila 1   :", A[1])          # una fila: un vector de 3
print("columna 2:", A[:, 2])       # una columna: un vector de 2
print("A[1, 2]  :", A[1, 2])       # fila 1, columna 2
print("Aᵀ forma :", A.T.shape)     # transpuesta: (3, 2)
# <<<
