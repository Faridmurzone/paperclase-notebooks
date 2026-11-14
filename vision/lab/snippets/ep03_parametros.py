import torch.nn as nn

# >>>
conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)    # 3×32×32 → 16×32×32
densa = nn.Linear(3 * 32 * 32, 16 * 32 * 32, device="meta")  # mismo mapeo
n_conv = sum(p.numel() for p in conv.parameters())
n_densa = sum(p.numel() for p in densa.parameters())
print(f"conv : {n_conv:>12,}")
print(f"densa: {n_densa:>12,}")
print(f"la densa tiene {n_densa / n_conv:,.0f} veces más")
# <<<
