import numpy as np
# >>>
def escalon(z):  return (z > 0).astype(float)
def sigmoide(z): return 1 / (1 + np.exp(-z))
def relu(z):     return np.maximum(0, z)

z = np.array([-2.0, 0.0, 0.6, 3.0])   # 0.6 es nuestra z
print("z        ", z)
print("escalón  ", escalon(z))
print("sigmoide ", sigmoide(z).round(2))
print("ReLU     ", relu(z))
# <<<
