import numpy as np
# >>>
x = np.array([1.0, 2.0, 3.0])    # 3 entradas
w = np.array([0.5, -0.5, 0.2])   # un peso por entrada
b = 0.5                          # el sesgo

# a mano: cada entrada por su peso, y sumar
z = 0.0
for i in range(3):
    print(f"x{i+1} * w{i+1} = {x[i]} * {w[i]:+} = {x[i]*w[i]:+.1f}")
    z += x[i] * w[i]
print(f"suma = {z:.1f}   + b = {z + b:.1f}")

# lo mismo en una línea: producto punto
z = w @ x + b
print("w @ x + b =", round(z, 2))
# <<<
