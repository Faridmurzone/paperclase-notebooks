import numpy as np
rng = np.random.default_rng(0)

# >>>
def paso_adam(gs, b1=0.9, b2=0.999):   # largo del paso, con lr = 1
    m = v = 0.0
    for t, g in enumerate(gs, start=1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
    return abs(m / (1 - b1 ** t)) / np.sqrt(v / (1 - b2 ** t))

n = 200
for nombre, gs in [("constante 0,001", np.full(n, 0.001)),
                   ("constante 1000 ", np.full(n, 1000.0)),
                   ("señal + ruido  ", 1 + 3 * rng.normal(size=n)),
                   ("solo ruido     ", 3 * rng.normal(size=n))]:
    print(f"{nombre}: paso {paso_adam(gs):.3f}")
# <<<
