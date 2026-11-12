import numpy as np
rng = np.random.default_rng(0)
# >>>
n, q = 20000, 0.84                        # q: cada persona coincide con el gusto común
comun = rng.random(n) < 0.5               # la preferencia de fondo en cada par
def etiquetar():                          # una persona: acierta el gusto común con prob. q
    return np.where(rng.random(n) < q, comun, ~comun)

a, b = etiquetar(), etiquetar()           # dos personas, los mismos pares
print(f"acuerdo entre dos personas:   {(a == b).mean():.3f}")
print(f"RM perfecto contra la persona a: {(comun == a).mean():.3f}")
print(f"RM que imita a la persona b:  {(b == a).mean():.3f}")
# <<<
