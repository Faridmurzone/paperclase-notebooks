import numpy as np
# >>>
A = np.array([[2, -1],
              [1,  1]])
x = np.array([1, 2])

# vista 1: cada fila de A, producto punto con x
for i in range(2):
    print(f"fila {i} · x = {A[i]} · {x} = {A[i] @ x}")

# vista 2: columnas de A, pesadas por las coordenadas de x
y = x[0] * A[:, 0] + x[1] * A[:, 1]
print("1·col0 + 2·col1 =", y)

print("A @ x =", A @ x)
# <<<
