import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(4)


def datos(n, T=10, V=10):
    """Tarea: para cada token, cuántas veces MÁS aparece en la secuencia (0..9)."""
    x = torch.randint(0, V, (n, T))
    return x, (x[:, :, None] == x[:, None, :]).sum(-1) - 1


def pila(pre, N=12, d=32):
    torch.manual_seed(0)
    capa = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=pre)
    enc = nn.TransformerEncoder(capa, N, enable_nested_tensor=False, norm=nn.LayerNorm(d) if pre else None)
    return nn.Sequential(nn.Embedding(10, d), enc, nn.Linear(d, 10))


# >>>
def entrenar(pre, pasos=200, lr=1e-3):          # 12 capas, sin warmup
    m = pila(pre)
    opt = torch.optim.Adam(m.parameters(), lr=lr, betas=(0.9, 0.98))
    torch.manual_seed(1)
    for _ in range(pasos):
        x, y = datos(64)
        loss = F.cross_entropy(m(x).flatten(0, 1), y.flatten())
        opt.zero_grad(); loss.backward(); opt.step()
    x, y = datos(2000)
    acc = (m(x).argmax(-1) == y).float().mean()
    return loss.item(), acc.item(), (y == 0).float().mean().item()

for pre in (False, True):
    loss, acc, base = entrenar(pre)
    print("pre " if pre else "post", f"pérdida {loss:.2f} · acierto {acc:.0%}")
print(f"decir siempre «0»: acierto {base:.0%}")
# <<<
