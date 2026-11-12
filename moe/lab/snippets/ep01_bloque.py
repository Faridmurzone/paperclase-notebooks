import torch.nn as nn
# >>>
d = 1024                                     # ancho del residual stream
atencion = nn.MultiheadAttention(d, num_heads=16, bias=False)
mlp = nn.Sequential(nn.Linear(d, 4 * d, bias=False),   # sube a 4d
                    nn.GELU(),
                    nn.Linear(4 * d, d, bias=False))   # baja a d
n_at = sum(p.numel() for p in atencion.parameters())
n_mlp = sum(p.numel() for p in mlp.parameters())
total = n_at + n_mlp
print(f"atención: {n_at:>10,}  = 4·d²  ({n_at / total:.0%})")
print(f"MLP:      {n_mlp:>10,}  = 8·d²  ({n_mlp / total:.0%})")
# <<<
