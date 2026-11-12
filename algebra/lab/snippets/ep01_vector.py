import numpy as np
# >>>
a = np.array([3.0, 1.0])     # un vector de 2 números
b = np.array([1.0, 2.0])

print("a      =", a, " forma:", a.shape)
print("a + b  =", a + b)     # coordenada a coordenada
print("2 * a  =", 2 * a)     # cada coordenada por 2
print("a[0]   =", a[0])      # la primera coordenada

largo = np.sqrt(a[0]**2 + a[1]**2)   # Pitágoras
print("‖a‖    =", round(largo, 2))
print("norm   =", round(np.linalg.norm(a), 2))
# <<<
