import math
# >>>
def escala(beta2, chicos=(10, 50, 200)):
    v, out = 0.0, []
    g = [1.0] * 1000 + [0.1] * max(chicos)    # el gradiente cae 10×
    for t, gi in enumerate(g, 1):
        v = beta2 * v + (1 - beta2) * gi ** 2  # promedio de g², como Adam
        v_hat = v / (1 - beta2 ** t)           # corrección de sesgo
        if t - 1000 in chicos:
            out.append(f"{math.sqrt(v_hat):.2f}")
    return out

for b2 in (0.98, 0.999):
    print(f"β2 = {b2:<5} (memoria ≈ {1 / (1 - b2):4.0f} pasos):",
          "√v̂ tras 10, 50, 200:", *escala(b2))
print("escala real del gradiente: 0.10")
# <<<
