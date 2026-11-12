import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
from transformers.utils import logging; logging.set_verbosity_error()
torch.manual_seed(0)
V, P, N = 16, 2, 5
w_oculto = torch.randn(V)                              # el mismo gusto sintético de ep04_rm.py
class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cuerpo = GPT2Model(GPT2Config(vocab_size=V, n_positions=16, n_embd=64, n_layer=2, n_head=4))
        self.cabeza = nn.Linear(64, 1)
    def forward(self, ids):
        return self.cabeza(self.cuerpo(input_ids=ids).last_hidden_state[:, -1]).squeeze(-1)
rm = RewardModel().eval()
rm.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep04_rm.pt")))
buenos = w_oculto.topk(5).indices                       # "demostraciones": respuestas escritas por personas
x = torch.randint(2, V, (200, P))
demos = torch.cat([x, buenos[torch.randint(0, 5, (200, N))]], 1)
y = torch.randint(2, V, (4, N)); par = torch.cat([x[:1].expand(4, P), y], 1)
# >>>
with torch.no_grad():
    r_demos = rm(demos)                       # el RM puntúa las demostraciones
    b = r_demos.mean()                        # el sesgo a restar
    print(f"antes:   promedio en demos {r_demos.mean():+.3f}")
    print(f"después: promedio en demos {(r_demos - b).mean().abs():.3f}")

    r = rm(par)                               # un prompt con 4 respuestas
    puesto = torch.tensor([0, 1, 2, 3])
    gana = puesto[:, None] < puesto[None, :]
    for nombre, rr in [("r", r), ("r − b", r - b)]:
        loss = -F.logsigmoid((rr[:, None] - rr[None, :])[gana]).mean()
        print(f"loss con {nombre:5}: {loss:.4f}")
# <<<
