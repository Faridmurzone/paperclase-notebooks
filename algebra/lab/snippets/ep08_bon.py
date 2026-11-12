import math
import torch
# >>>
N = 100_000                                # respuestas, ordenadas por recompensa
i = torch.arange(1, N + 1, dtype=torch.float64)
ref = torch.full((N,), 1 / N, dtype=torch.float64)  # el modelo base: uniforme

for n in [1, 4, 16, 64]:
    # best-of-n: sortear n y quedarse con la de mayor recompensa
    bon = (i / N) ** n - ((i - 1) / N) ** n
    kl = (bon * torch.log(bon / ref)).sum().item()
    formula = math.log(n) - (n - 1) / n
    print(f"n={n:3}: KL medido {round(kl, 3) + 0.0:.3f}   log n - (n-1)/n = {formula:.3f}")
# <<<
