import torch.nn as nn
d_model, T = 512, 100
# >>>
for h in [1, 8]:
    mha = nn.MultiheadAttention(d_model, h, bias=False)
    params = sum(p.numel() for p in mha.parameters())
    d_k = d_model // h
    mults = h * (T * T * d_k) * 2      # Q·Kᵀ y A·V, todas las heads
    print(f"h={h}  d_k={d_k:3}  parámetros={params:,}"
          f"  multiplicaciones (T=100)={mults:,}")
# <<<
