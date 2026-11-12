import torch
import torch.nn as nn
torch.manual_seed(0)
V, d, BOS = 10, 64, 10          # dígitos 0–9 y un token de inicio


class Seq2Seq(nn.Module):       # encoder y decoder GRU, sin atención
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V + 1, d)
        self.enc = nn.GRU(d, d, batch_first=True)
        self.dec = nn.GRU(d, d, batch_first=True)
        self.out = nn.Linear(d, V)

    def forward(self, x, y_in):
        _, h = self.enc(self.emb(x))           # h: el único puente
        s, _ = self.dec(self.emb(y_in), h)
        return self.out(s)


def entrenar_y_medir(L, pasos=600):
    m = Seq2Seq()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(pasos):
        x = torch.randint(0, V, (128, L))
        y = x.flip(1)
        y_in = torch.cat([torch.full((128, 1), BOS), y[:, :-1]], 1)
        loss = nn.functional.cross_entropy(m(x, y_in).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    x = torch.randint(0, V, (1000, L)); y = x.flip(1)
    y_in = torch.cat([torch.full((1000, 1), BOS), y[:, :-1]], 1)
    with torch.no_grad():
        return (m(x, y_in).argmax(-1) == y).float().mean().item()
# >>>
# misma red, mismo entrenamiento: invertir secuencias de L dígitos
for L in [4, 8, 16, 32]:
    acierto = entrenar_y_medir(L)          # 600 pasos de Adam
    print(f"L = {L:2}: aciertos por dígito = {acierto:.0%}")
# <<<
