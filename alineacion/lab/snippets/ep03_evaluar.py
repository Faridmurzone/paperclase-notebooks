import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel
torch.manual_seed(0)
V, PAD = 16, 0
w_oculto = torch.randn(V); w_oculto[PAD] = 0          # "gusto" oculto de quien etiqueta, por token
def r_oculta(ids):                                     # r*(y) = suma de w* sobre los tokens
    return w_oculto[ids].sum(-1) * 0.7

def respuestas(n):                                     # longitudes 3..8, relleno a la derecha
    L = torch.randint(3, 9, (n,))
    ids = torch.randint(2, V, (n, 8))
    mask = (torch.arange(8) < L[:, None]).long()
    return ids * mask, mask

def pares(n):                                          # dos respuestas y quién ganó según Bradley-Terry
    a, ma = respuestas(n); b, mb = respuestas(n)
    p_a = torch.sigmoid(r_oculta(a) - r_oculta(b))
    gana_a = torch.rand(n) < p_a
    w = torch.where(gana_a[:, None], a, b); mw = torch.where(gana_a[:, None], ma, mb)
    l = torch.where(gana_a[:, None], b, a); ml = torch.where(gana_a[:, None], mb, ma)
    return w, mw, l, ml, p_a

cfg = LlamaConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=4, pad_token_id=PAD)
class ModeloRecompensa(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cuerpo = LlamaModel(cfg)                   # transformer sin la capa de logits
        self.cabeza = nn.Linear(cfg.hidden_size, 1)     # un solo número
    def forward(self, ids, mask):
        h = self.cuerpo(input_ids=ids, attention_mask=mask).last_hidden_state
        ultimo = mask.sum(1) - 1                        # índice del último token real
        h_ult = h[torch.arange(len(ids)), ultimo]       # [batch, d]
        return self.cabeza(h_ult).squeeze(-1)           # [batch]

def entrenar(rm, pasos=600):
    opt = torch.optim.AdamW(rm.parameters(), lr=1e-3)
    for _ in range(pasos):
        w, mw, l, ml, _ = pares(64)
        loss = -F.logsigmoid(rm(w, mw) - rm(l, ml)).mean()
        loss.backward(); opt.step(); opt.zero_grad()
    return rm
rm = entrenar(ModeloRecompensa(cfg))
torch.manual_seed(1)
# >>>
w, mw, l, ml, p_a = pares(4000)                     # pares nuevos, nunca vistos
with torch.no_grad():
    acc = (rm(w, mw) > rm(l, ml)).float().mean()
techo = torch.maximum(p_a, 1 - p_a).mean()   # apostar al más probable
print(f"accuracy del RM:   {acc:.3f}")
print(f"techo alcanzable:  {techo:.3f}")

y, m = respuestas(4000)
with torch.no_grad():
    r = rm(y, m)
c = torch.corrcoef(torch.stack([r, r_oculta(y)]))[0, 1]
print(f"correlación con r*: {c:.3f}")
print(f"media de r: {r.mean():+.2f}  (la escala absoluta no importa)")
# <<<
