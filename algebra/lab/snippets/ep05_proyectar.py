import numpy as np
# >>>
x = np.array([2.0, 1.0])
r = np.array([3.0, 4.0])              # una dirección cualquiera

r_hat = r / np.linalg.norm(r)         # largo 1
print("r̂ =", r_hat)

c = x @ r_hat                          # cuánto de x va hacia r̂
sombra = c * r_hat                     # la componente sobre r̂
x_prima = x - sombra                   # lo que queda
print("x·r̂ =", round(c, 2), "  sombra =", sombra.round(2))
print("x' =", x_prima.round(2))
print("x'·r̂ =", round(x_prima @ r_hat, 6) + 0.0)   # 0: perpendicular
print("sombra + x' =", (sombra + x_prima).round(2))
# <<<
