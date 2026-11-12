import numpy as np
x = np.array([1.0, 2.0, 3.0])
# >>>
W = np.array([[ 0.5, -0.5, 0.2],    # fila 0 = neurona 1
              [-0.3,  0.1, 0.4]])   # fila 1 = neurona 2
b = np.array([0.5, 0.0])            # un sesgo por neurona

z = W @ x + b                       # las dos de una vez
print("W:", W.shape, " x:", x.shape, " z:", z.shape)
print("z =", z.round(2))
# <<<
