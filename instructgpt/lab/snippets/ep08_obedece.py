import numpy as np
rng = np.random.default_rng(0)
# >>>
# 4 respuestas candidatas a «escribí algo agresivo sobre X»
#            [cumple el pedido, daño]
candidatas = np.array([[1.0, 0.9],          # obedece
                       [0.6, 0.4],          # obedece a medias
                       [0.2, 0.0],          # se niega con amabilidad
                       [0.0, 0.0]])         # ignora el pedido
rm_util = np.array([2.0, -0.2])             # entrenado priorizando ser útil
rm_cauto = np.array([2.0, -3.0])            # si el daño pesara de verdad
for nombre, w in [("RM útil", rm_util), ("RM cauto", rm_cauto)]:
    puntajes = candidatas @ w
    elegida = puntajes.argmax()             # best-of-4: la que el RM prefiere
    print(f"{nombre:8} puntajes {puntajes.round(1) + 0.0} → elige {elegida}"
          f" (daño {candidatas[elegida, 1]})")
# <<<
