import numpy as np
rng = np.random.default_rng(6)
# >>>
n = 400                                   # prompts de test: un juicio por prompt
p_oculta = 0.66                           # cuánto gusta A de verdad (no se ve)
prefiere_A = rng.random(n) < p_oculta     # True: quien etiqueta eligió A sobre B
print("primeros 12:", prefiere_A[:12].astype(int))

win_rate = prefiere_A.mean()              # promedio de unos y ceros
print(f"A gana {prefiere_A.sum()} de {n}: win rate {win_rate:.3f}")

contra_si = rng.random(n) < 0.5           # el baseline contra sí mismo
print(f"SFT contra SFT: {contra_si.mean():.3f}")
# <<<
