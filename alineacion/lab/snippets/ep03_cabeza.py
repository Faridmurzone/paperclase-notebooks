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
# >>>
class ModeloRecompensa(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cuerpo = LlamaModel(cfg)           # transformer sin logits
        self.head = nn.Linear(cfg.hidden_size, 1)    # reward head
    def forward(self, ids, mask):
        h = self.cuerpo(input_ids=ids, attention_mask=mask).last_hidden_state
        ultimo = mask.sum(1) - 1                # último token real
        h_ult = h[torch.arange(len(ids)), ultimo]   # [batch, d]
        return self.head(h_ult).squeeze(-1)     # [batch]

rm = ModeloRecompensa(cfg)
ids, mask = respuestas(3)
print(mask)
print("r:", rm(ids, mask).detach().numpy().round(3))
mas_relleno = torch.cat([ids, torch.zeros(3, 4, dtype=torch.long)], 1)
mask2 = torch.cat([mask, torch.zeros(3, 4, dtype=torch.long)], 1)
print("con más relleno:", rm(mas_relleno, mask2).detach().numpy().round(3))
# <<<
