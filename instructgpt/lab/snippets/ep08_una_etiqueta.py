import numpy as np
rng = np.random.default_rng(0)
# >>>
n = 6000
claro = rng.random(n) < 0.5               # la mitad de los pares: casi todos piensan igual
p_A = np.where(claro, 0.95, 0.60)         # prob. de que una persona elija A

una = rng.random(n) < p_A                 # el paper: una etiqueta por par
tres = (rng.random((n, 3)) < p_A[:, None]).sum(1)   # tres personas
dividido = (tres == 1) | (tres == 2)

print("con una etiqueta, desacuerdo visible: ninguno")
print(f"con tres: divididos {dividido.mean():.0%} de los pares")
print(f"  entre los claros {dividido[claro].mean():.0%},"
      f" entre los polémicos {dividido[~claro].mean():.0%}")
# <<<
