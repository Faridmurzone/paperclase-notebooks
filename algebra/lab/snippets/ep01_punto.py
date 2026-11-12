import numpy as np
import torch
# >>>
a = np.array([3.0, 1.0])
b = np.array([1.0, 2.0])

# a mano: multiplicar coordenada a coordenada y sumar
total = 0.0
for i in range(len(a)):
    print(f"a[{i}]*b[{i}] = {a[i]} * {b[i]} = {a[i]*b[i]}")
    total += a[i] * b[i]
print("suma      =", total)

print("a @ b     =", a @ b)            # NumPy
print("np.dot    =", np.dot(a, b))
print("torch.dot =", torch.dot(torch.tensor(a), torch.tensor(b)).item())
# <<<
