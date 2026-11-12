import math
# >>>
def lrate(paso, d_model=512, warmup=4000):          # ecuación (3)
    return d_model ** -0.5 * min(paso ** -0.5, paso * warmup ** -1.5)

for paso in [1, 1000, 4000, 16000, 64000, 100000]:
    print(f"paso {paso:6}: lr = {lrate(paso):.2e}")

pico = (512 * 4000) ** -0.5                          # 1 / √(d · w)
forma = lambda s, w=4000: min(s / w, math.sqrt(w / s))
print(f"pico {pico:.2e} · igual a pico · forma:",
      all(abs(lrate(s) - pico * forma(s)) < 1e-12 for s in range(1, 10**5)))
# <<<
