import numpy as np
rng = np.random.default_rng(0)
C = rng.normal(0, 1, size=(1000, 2)); C = C - C.mean(0)
def grad(p, idx): return 2 * (p - C[idx]).mean(0)
p = np.array([-2.5, 1.5])

# >>>
for B in [1, 4, 16, 64]:
    gs = np.array([grad(p, rng.choice(1000, B, replace=False))
                   for _ in range(4000)])          # 4000 minibatches
    error = gs - grad(p, np.arange(1000))          # ruido: g_B - ∇L
    print(f"B = {B:2}: sesgo {error.mean(0).round(3)}"
          f"  desvío {error.std():.3f}"
          f"  desvío·√B {error.std() * np.sqrt(B):.2f}")
# <<<
