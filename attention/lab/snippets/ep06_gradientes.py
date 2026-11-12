import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(4)


def datos(n, T=10, V=10):
    """Tarea: para cada token, cuántas veces MÁS aparece en la secuencia (0..9)."""
    x = torch.randint(0, V, (n, T))
    return x, (x[:, :, None] == x[:, None, :]).sum(-1) - 1


# >>>
def pila(pre, N=12, d=32):
    torch.manual_seed(0)                        # misma init para las dos
    capa = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0,
                                      batch_first=True, norm_first=pre)
    enc = nn.TransformerEncoder(capa, N, enable_nested_tensor=False,
                                norm=nn.LayerNorm(d) if pre else None)
    return nn.Sequential(nn.Embedding(10, d), enc, nn.Linear(d, 10))

torch.manual_seed(1); x, y = datos(256)
for pre in (False, True):
    m = pila(pre)
    F.cross_entropy(m(x).flatten(0, 1), y.flatten()).backward()
    g = [c.linear1.weight.grad.norm().item() for c in m[1].layers]
    print("pre " if pre else "post", " ".join(f"{v:.2f}" for v in g))
# <<<
