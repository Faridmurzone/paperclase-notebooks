import math
import os
import torch
import torch.nn as nn
V, BOS, L, d = 10, 10, 16, 64
GUARDADO = os.path.join(os.path.dirname(__file__), "..", "datos", "ep05_inversor.pt")


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


modelo = Inversor()
modelo.load_state_dict(torch.load(GUARDADO))   # lo entrena ep05_entrenar.py
modelo.eval()
torch.manual_seed(7)
x = torch.randint(0, V, (1, L))
emb = lambda t: modelo.emb(t) * math.sqrt(d) + modelo.pe[:t.shape[1]]
mascara = nn.Transformer.generate_square_subsequent_mask
texto = lambda t: "".join(map(str, t.tolist()))
# >>>
memoria = modelo.tr.encoder(emb(x))          # el encoder corre UNA vez
y = torch.tensor([[BOS]])                    # el decoder arranca con BOS
with torch.no_grad():
    for t in range(L):
        causal = mascara(y.shape[1])               # triangular
        h = modelo.tr.decoder(emb(y), memoria, tgt_mask=causal)
        siguiente = modelo.out(h[:, -1]).argmax(-1)   # solo el último
        y = torch.cat([y, siguiente.unsqueeze(0)], 1)
        if t in (0, 1, 2, L - 1):
            print(f"paso {t + 1:2}: {texto(y[0, 1:])}")
print("entrada :", texto(x[0]))
print("al revés:", texto(x[0].flip(0)))
# <<<
