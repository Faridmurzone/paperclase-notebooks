import numpy as np
# >>>
relu = lambda z: np.maximum(0, z)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

W1 = np.array([[1, 1],
               [1, 1]])
b1 = np.array([0, -1])         # la segunda neurona se corre 1
w2 = np.array([1, -2])         # salida: h1 - 2·h2

H = relu(X @ W1.T + b1)        # hidden layer con ReLU
y = H @ w2
print("H =", H.tolist())
print("y =", y.tolist(), " (XOR: [0, 1, 1, 0])")
# <<<
