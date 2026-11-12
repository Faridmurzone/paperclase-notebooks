import torch
import torch.nn as nn
torch.manual_seed(0)
# >>>
d_model, d_ff = 512, 2048                     # los del paper
ffn = nn.Sequential(nn.Linear(d_model, d_ff),  # W1, b1: sube a 2048
                    nn.ReLU(),                 # max(0, ·)
                    nn.Linear(d_ff, d_model))  # W2, b2: baja a 512

x = torch.randn(2, 5, d_model)                # batch 2, 5 tokens
y = ffn(x)
print("entrada", tuple(x.shape), "→ salida", tuple(y.shape))

# cada token por separado: el token 3, solo, da lo mismo
solo = ffn(x[:, 3])
print("token 3 solo == token 3 en la frase:",
      torch.allclose(solo, y[:, 3], atol=1e-6))

n_ffn = sum(p.numel() for p in ffn.parameters())
n_att = 4 * d_model * d_model + 4 * d_model   # W^Q, W^K, W^V, W^O
print(f"parámetros FFN {n_ffn:,} · atención {n_att:,}")
# <<<
