import numpy as np
# >>>
x = np.array([1.0, 2.0, 3.0])              # las mismas 3 entradas

w1, b1 = np.array([0.5, -0.5, 0.2]), 0.5   # neurona 1 (la del ep. 1)
w2, b2 = np.array([-0.3, 0.1, 0.4]), 0.0   # neurona 2

z1 = w1 @ x + b1
z2 = w2 @ x + b2
print("z1 =", round(z1, 2), "  z2 =", round(z2, 2))
# <<<
