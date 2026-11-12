import numpy as np
# >>>
# la respuesta correcta es 1; ¿cuánto cuesta cada probabilidad?
for p in [0.9, 0.5, 0.1, 0.01]:
    mse = (1 - p) ** 2
    ce = -np.log(p)
    print(f"p = {p:<4}  MSE = {mse:.2f}   -log p = {ce:.2f}")
# <<<
