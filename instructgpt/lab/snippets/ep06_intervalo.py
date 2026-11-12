import numpy as np
rng = np.random.default_rng(6)
n = 400
prefiere_A = rng.random(n) < 0.66         # los mismos 400 juicios del snippet anterior
# >>>
p = prefiere_A.mean()
se = np.sqrt(p * (1 - p) / n)             # error estándar de un promedio de 0 y 1
print(f"p = {p:.3f}   se = {se:.4f}")
print(f"fórmula:   {p - 1.96*se:.3f} a {p + 1.96*se:.3f}  (± {1.96*se:.3f})")

remuestras = rng.choice(prefiere_A, size=(5000, n))   # bootstrap
medias = remuestras.mean(1)
lo, hi = np.percentile(medias, [2.5, 97.5])
print(f"bootstrap: {lo:.3f} a {hi:.3f}")
# <<<
