import numpy as np
# >>>
def n_para(p, margen, z=1.96):            # despejar n de  margen = z·√(p(1−p)/n)
    return z**2 * p * (1 - p) / margen**2

casos = {"175B vs GPT-3, 85 ± 3": (0.85, 0.03),
         "175B vs few-shot, 71 ± 4": (0.71, 0.04),
         "figura 1, 0,66 ± 0,045": (0.66, 0.045)}
for nombre, (p, m) in casos.items():
    print(f"{nombre:26} → n ≈ {n_para(p, m):4.0f}")
print(f"y para ± 1 punto con p = 0,5: n ≈ {n_para(0.5, 0.01):5.0f}")
# <<<
