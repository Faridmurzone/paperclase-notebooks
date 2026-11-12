import torch
import torch.nn as nn
torch.manual_seed(0)
# >>>
class Bloque(nn.Module):
    def __init__(self, d=64, cabezas=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, cabezas, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):                      # x: [batch, pos, d]
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask)[0]   # suma al residual stream
        x = x + self.mlp(self.ln2(x))                    # y vuelve a escribir
        return x

x = torch.randn(1, 5, 64)
print(Bloque()(x).shape)
# <<<
