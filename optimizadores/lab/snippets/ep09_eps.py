import torch

# >>>
eta, eps = 1.0, 1e-8
for g in [1.0, 1e-4, 1e-8, 1e-10]:                   # primer paso: m̂ = g, √v̂ = |g|
    paso = eta * g / (abs(g) + eps)
    print(f"g = {g:7.0e}   paso = {paso:.4f} · η")
print(f"1e-8 en float32: {torch.tensor(1e-8, dtype=torch.float32).item():.3g}")
print(f"1e-8 en float16: {torch.tensor(1e-8, dtype=torch.float16).item():.3g}")
# <<<
