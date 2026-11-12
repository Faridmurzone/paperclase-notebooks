import math
import os
import torch
import torch.nn as nn
torch.manual_seed(0)
torch.set_num_threads(4)
V, BOS, L, d = 10, 10, 16, 64          # dígitos 0–9, BOS = 10, largo 16
GUARDAR = os.path.join(os.path.dirname(__file__), "..", "datos", "ep05_inversor.pt")


def pe(n, d):
    p = torch.arange(n).unsqueeze(1); om = 1 / 10000 ** (torch.arange(0, d, 2) / d)
    out = torch.zeros(n, d); out[:, 0::2], out[:, 1::2] = torch.sin(p * om), torch.cos(p * om)
    return out


class Inversor(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V + 1, d)
        self.register_buffer("pe", pe(64, d))
        self.tr = nn.Transformer(d, nhead=4, num_encoder_layers=2, num_decoder_layers=2,
                                 dim_feedforward=128, dropout=0.0, batch_first=True)
        self.out = nn.Linear(d, V)

    def forward(self, x, y_in):
        T = y_in.shape[1]
        causal = nn.Transformer.generate_square_subsequent_mask(T)
        h = self.tr(self.emb(x) * math.sqrt(d) + self.pe[:x.shape[1]],
                    self.emb(y_in) * math.sqrt(d) + self.pe[:T], tgt_mask=causal)
        return self.out(h)


def lote(n):
    x = torch.randint(0, V, (n, L)); y = x.flip(1)
    return x, y, torch.cat([torch.full((n, 1), BOS), y[:, :-1]], 1)
# >>>
modelo = Inversor()
opt = torch.optim.Adam(modelo.parameters(), lr=2e-3)
for paso in range(1, 801):
    x, y, y_in = lote(128)                 # y_in = BOS + y corrida 1
    logits = modelo(x, y_in)               # [128, 16, 10]
    loss = nn.functional.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if paso in (1, 200, 400, 600, 800):
        print(f"paso {paso:3}: pérdida {loss.item():.3f}")

x, y, y_in = lote(1000)
with torch.no_grad():
    acierto = (modelo(x, y_in).argmax(-1) == y).float().mean()
print(f"L = {L}: aciertos por dígito = {acierto:.1%}")
# <<<
torch.save(modelo.state_dict(), GUARDAR)
