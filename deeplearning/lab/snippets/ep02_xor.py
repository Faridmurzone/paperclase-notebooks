import numpy as np
# >>>
escalon = lambda z: (z > 0).astype(int)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])   # las 4 entradas: un batch

W1 = np.array([[1, 1],      # neurona h1: OR
               [1, 1]])     # neurona h2: AND
b1 = np.array([-0.5, -1.5])
w2, b2 = np.array([1, -1]), -0.5                 # salida: OR y no AND

H = escalon(X @ W1.T + b1)     # hidden layer: (4, 2)
y = escalon(H @ w2 + b2)       # salida: (4,)
print("H =", H.tolist())
print("y =", y.tolist(), " (XOR: [0, 1, 1, 0])")
# <<<
