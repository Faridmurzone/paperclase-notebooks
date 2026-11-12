import numpy as np
# >>>
x = np.array([1.0, 2.0]);  y = 2.0
W1 = np.array([[0.5, -1.0], [1.0, 0.5]]);  b1 = np.zeros(2)
w2 = np.array([1.0, 0.5]);                 b2 = 0.0

# forward
z1 = W1 @ x + b1;  h = np.maximum(0, z1)
y_hat = w2 @ h + b2;  L = (y_hat - y) ** 2
print("z1 =", z1, " h =", h)
print("y_hat =", y_hat, " L =", L)

# backward
d = 2 * (y_hat - y)                 # dL/dy_hat
dw2, db2 = d * h, d                 # capa 2
dh = d * w2                         # vuelve a la hidden layer
dz1 = dh * (z1 > 0)                 # la ReLU deja pasar o corta
dW1, db1 = np.outer(dz1, x), dz1    # capa 1
print("dw2 =", dw2 + 0.0, " db2 =", db2)
print("dh  =", dh, " dz1 =", dz1 + 0.0)
print("dW1 =", (dW1 + 0.0).tolist())
# <<<
