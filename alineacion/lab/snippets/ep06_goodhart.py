import numpy as np
rng = np.random.default_rng(0)
# >>>
def mejor_de_n(n, ruido, rondas=2000):
    U = rng.normal(size=(rondas, n))              # lo que de verdad queremos
    P = U + ruido(size=(rondas, n))               # el proxy que medimos
    elegida = P.argmax(axis=1)                    # optimizar: el máximo
    fila = np.arange(rondas)
    return P[fila, elegida].mean(), U[fila, elegida].mean()

normal = lambda size: rng.normal(size=size)
colas = lambda size: rng.standard_t(df=1.5, size=size)  # errores enormes
print("    n | proxy  U (ruido normal) | proxy  U (colas pesadas)")
for n in [1, 16, 256, 4096]:
    pn, un = mejor_de_n(n, normal)
    pc, uc = mejor_de_n(n, colas)
    print(f"{n:5d} | {pn:5.2f} {un:5.2f}            | {pc:6.1f} {uc:5.2f}")
# <<<
