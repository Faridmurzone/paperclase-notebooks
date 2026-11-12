import numpy as np
# >>>
x = np.array([4.0, 3.0, 0.0])

l2 = np.sqrt(np.sum(x ** 2))          # Pitágoras
l1 = np.sum(np.abs(x))                # sumar valores absolutos
l0 = np.count_nonzero(x)              # contar los que no son cero
print("L2 =", l2, "  L1 =", l1, "  L0 =", l0)

print("np.linalg.norm:", np.linalg.norm(x, 2),
      np.linalg.norm(x, 1), np.linalg.norm(x, 0))
print("L2 al cuadrado =", x @ x)      # producto punto consigo mismo
# <<<
