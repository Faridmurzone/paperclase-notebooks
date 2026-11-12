import numpy as np
rng = np.random.default_rng(0)
# >>>
peso = {"grupo A": 0.8, "grupo B": 0.2}   # cuánta gente de cada grupo etiqueta
prefiere_X = {"grupo A": 0.75, "grupo B": 0.15}   # estilo X contra estilo Y

n = 10000
grupo = np.where(rng.random(n) < peso["grupo A"], "grupo A", "grupo B")
p = np.vectorize(prefiere_X.get)(grupo)
elige_X = rng.random(n) < p
print(f"todas las etiquetas juntas: X gana {elige_X.mean():.2f}")
for g in peso:                            # el modelo que siempre responde X
    print(f"  win rate de X para el {g}: {elige_X[grupo == g].mean():.2f}")
# <<<
