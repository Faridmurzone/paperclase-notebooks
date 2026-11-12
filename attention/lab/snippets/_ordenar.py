"""Tarea de juguete de los episodios 7 y 8: ordenar L dígitos. Modelo: nn.Transformer chico (post-norm, como el paper).
No es un snippet (empieza con _ y correr_snippets.py no lo corre); lo importan ep07_* y ep08_* desde su setup."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

V, BOS, L = 10, 10, 10                      # dígitos 0–9, BOS = 10, largo 10
torch.set_num_threads(1)


def pe(n, d):
    p = torch.arange(n).unsqueeze(1); om = 1 / 10000 ** (torch.arange(0, d, 2) / d)
    out = torch.zeros(n, d); out[:, 0::2], out[:, 1::2] = torch.sin(p * om), torch.cos(p * om)
    return out


def datos(n, gen=None, tarea="ordenar"):
    x = torch.randint(0, V, (n, L), generator=gen)
    y = x.sort(1).values if tarea == "ordenar" else x.flip(1)
    return x, y, torch.cat([torch.full((n, 1), BOS), y[:, :-1]], 1)


class Ordenador(nn.Module):
    def __init__(self, d=32, h=4, capas=2, d_ff=128, p=0.1, pe_tipo="seno"):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(V + 1, d)
        if pe_tipo == "seno":
            self.register_buffer("pos", pe(32, d))
        elif pe_tipo == "aprendido":
            self.pos = nn.Parameter(torch.randn(32, d) * 0.02 * math.sqrt(d))
        else:
            self.register_buffer("pos", torch.zeros(32, d))
        self.drop = nn.Dropout(p)
        self.tr = nn.Transformer(d, h, capas, capas, d_ff, dropout=p, batch_first=True)
        self.out = nn.Linear(d, V)

    def entrada(self, t):
        return self.drop(self.emb(t) * math.sqrt(self.d) + self.pos[:t.shape[1]])

    def forward(self, x, y_in):
        causal = nn.Transformer.generate_square_subsequent_mask(y_in.shape[1])
        return self.out(self.tr(self.entrada(x), self.entrada(y_in), tgt_mask=causal))


def forma(paso, warmup):
    """La forma de la ecuación (3): sube lineal hasta warmup y baja como 1/√paso. lr = pico · forma."""
    paso = max(paso, 1)
    return min(paso / warmup, math.sqrt(warmup / paso))


def entrenar(modelo, pasos=800, pico=2e-3, warmup=100, eps_ls=0.1, batch=64, semilla=1, tarea="ordenar", fijos=None):
    """fijos = n: entrena siempre sobre las mismas n secuencias (para ver overfitting); None: secuencias nuevas en cada paso."""
    torch.manual_seed(semilla)
    opt = torch.optim.Adam(modelo.parameters(), lr=pico, betas=(0.9, 0.98), eps=1e-9)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: forma(s + 1, warmup))
    banco = datos(fijos, torch.Generator().manual_seed(7), tarea) if fijos else None
    modelo.train()
    for paso in range(1, pasos + 1):
        if fijos:
            i = torch.randint(0, fijos, (batch,)); x, y, y_in = banco[0][i], banco[1][i], banco[2][i]
        else:
            x, y, y_in = datos(batch, tarea=tarea)
        loss = F.cross_entropy(modelo(x, y_in).flatten(0, 1), y.flatten(), label_smoothing=eps_ls)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return modelo


@torch.no_grad()
def evaluar(modelo, n=1000, tarea="ordenar"):
    """Con la salida correcta como entrada del decoder (teacher forcing): acierto por dígito, secuencias enteras bien,
    probabilidad media de su respuesta (confianza) y perplejidad = e^(pérdida sin suavizar)."""
    modelo.eval()
    x, y, y_in = datos(n, torch.Generator().manual_seed(123), tarea)
    p = modelo(x, y_in).softmax(-1)
    pred = p.argmax(-1)
    nll = -p.gather(-1, y.unsqueeze(-1)).log().mean()
    return dict(digito=(pred == y).float().mean().item(), secuencia=(pred == y).all(1).float().mean().item(),
                confianza=p.max(-1).values.mean().item(), perplejidad=nll.exp().item())
