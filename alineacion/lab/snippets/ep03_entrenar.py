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
vw, vmw, vl, vml, _ = pares(2000)                    # pares de validación
# >>>
rm = ModeloRecompensa(cfg)
opt = torch.optim.AdamW(rm.parameters(), lr=1e-3)
for paso in range(601):
    w, mw, l, ml, _ = pares(64)                     # elegida, rechazada
    r_w, r_l = rm(w, mw), rm(l, ml)                 # dos forwards, mismo modelo
    loss = -F.logsigmoid(r_w - r_l).mean()          # L_RM = −E log σ(Δr)
    loss.backward(); opt.step(); opt.zero_grad()
    if paso % 200 == 0:
        with torch.no_grad():                       # ¿ordena bien pares nuevos?
            acc = (rm(vw, vmw) > rm(vl, vml)).float().mean()
        print(f"paso {paso:3d}  loss {loss.item():.3f}  acc {acc:.2f}")
# <<<
