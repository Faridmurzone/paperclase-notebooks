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
# >>>
def pares(n):                          # dos respuestas por prompt, y quién ganó
    a, ma = respuestas(n)
    b, mb = respuestas(n)
    p_a = torch.sigmoid(r_oculta(a) - r_oculta(b))   # Bradley-Terry
    gana_a = torch.rand(n) < p_a                     # la persona elige
    return a, b, p_a, gana_a

a, b, p_a, gana_a = pares(4)
for i in range(4):
    print(f"r*(a)={r_oculta(a[i]):+.2f} r*(b)={r_oculta(b[i]):+.2f}"
          f"  P(a≻b)={p_a[i]:.2f}  ganó {'a' if gana_a[i] else 'b'}")
# <<<
