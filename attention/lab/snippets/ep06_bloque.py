import torch
import torch.nn as nn
torch.manual_seed(0)


def copiar(a, r):
    """Copia los pesos de nn.TransformerEncoderLayer (r) a nuestro Bloque (a)."""
    a.attn.load_state_dict(r.self_attn.state_dict())
    for x, y in [(a.ffn[0], r.linear1), (a.ffn[2], r.linear2), (a.ln1, r.norm1), (a.ln2, r.norm2)]:
        x.load_state_dict(y.state_dict())


# >>>
class Bloque(nn.Module):
    def __init__(s, d=512, h=8, d_ff=2048, p=0.1, pre=False):
        super().__init__()
        s.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        s.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(), nn.Linear(d_ff, d))
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.drop, s.pre = nn.Dropout(p), pre

    def forward(s, x):
        if s.pre:                                   # pre-norm (hoy)
            h = s.ln1(x)
            x = x + s.drop(s.attn(h, h, h)[0])
            return x + s.drop(s.ffn(s.ln2(x)))
        x = s.ln1(x + s.drop(s.attn(x, x, x)[0]))   # post-norm (paper)
        return s.ln2(x + s.drop(s.ffn(x)))

ref = nn.TransformerEncoderLayer(512, 8, 2048, batch_first=True)
nuestro = Bloque(); copiar(nuestro, ref)            # mismos pesos
nuestro.eval(); ref.eval()                          # sin dropout
x = torch.randn(2, 7, 512)
print("dif. máx con PyTorch:", f"{(nuestro(x) - ref(x)).abs().max():.1e}")
# <<<
