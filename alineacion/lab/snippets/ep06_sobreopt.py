import numpy as np
rng = np.random.default_rng(0)
D = 8
w = rng.normal(size=D) / np.sqrt(D)
oro = lambda X: X @ w - 0.15 * (X ** 2).sum(-1)       # "humano" sintético
X0 = rng.normal(size=(300, D))                        # 300 etiquetas ruidosas
y0 = oro(X0) + rng.normal(scale=0.5, size=300)
coef = np.linalg.lstsq(np.c_[X0, np.ones(300)], y0, rcond=None)[0]
proxy = lambda X: X @ coef[:-1] + coef[-1]            # RM lineal ajustado
filas = []
# >>>
for n in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    X = rng.normal(size=(2000, n, D))         # n respuestas de π_init
    xs = X[np.arange(2000), proxy(X).argmax(1)]   # best-of-n según el proxy
    kl = np.log(n) - (n - 1) / n              # KL exacto de best-of-n
    filas.append((np.sqrt(kl), proxy(xs).mean(), oro(xs).mean()))
d, r_proxy, r_oro = np.array(filas).T
r_proxy, r_oro = r_proxy - r_proxy[0], r_oro - r_oro[0]   # R(0) = 0
A = np.stack([d, -d ** 2], axis=1)            # R(d) = αd − βd²
alfa, beta = np.linalg.lstsq(A, r_oro, rcond=None)[0]
for i in [0, 3, 6, 9]:
    print(f"d={d[i]:.2f}  proxy {r_proxy[i]:+.2f}  oro {r_oro[i]:+.2f}")
print(f"α={alfa:.2f}  β={beta:.2f}  pico en d*=α/2β = {alfa/(2*beta):.2f}")
# <<<
