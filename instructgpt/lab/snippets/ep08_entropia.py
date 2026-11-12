import numpy as np
# >>>
def entropia(p):                           # en bits: 1 = no elige, 0 = segurísimo
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

# log-probabilidades (inventadas) de las dos frases de un par
for lp1, lp2 in [(-20.0, -20.0), (-20.0, -21.0), (-20.0, -23.0)]:
    p1 = np.exp(lp1) / (np.exp(lp1) + np.exp(lp2))   # P(frase 1 | par)
    print(f"log p: {lp1:.0f} y {lp2:.0f} → p = {p1:.2f}, entropía {entropia(p1):.2f}")

# Tabla 14, CrowS-Pairs, 175B: entropía (más alta = menos sesgo)
print("GPT básico 0.410 | PPO-ptx básico 0.413 | PPO-ptx respetuoso 0.243")
# <<<
