import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
from transformers.utils import logging; logging.set_verbosity_error()
torch.manual_seed(0)
cfg = GPT2Config(vocab_size=12, n_positions=16, n_embd=64, n_layer=2, n_head=4)
class RewardModel(nn.Module):                         # el RM del episodio 4
    def __init__(self):
        super().__init__()
        self.cuerpo, self.cabeza = GPT2Model(cfg), nn.Linear(64, 1)
    def forward(self, ids):
        return self.cabeza(self.cuerpo(input_ids=ids).last_hidden_state[:, -1]).squeeze(-1)
rm = RewardModel().eval()
# >>>
class FuncionDeValor(nn.Module):              # V(prefijo): un número por posición
    def __init__(self):
        super().__init__()
        self.cuerpo, self.cabeza = GPT2Model(cfg), nn.Linear(64, 1)
    def forward(self, ids):
        h = self.cuerpo(input_ids=ids).last_hidden_state
        return self.cabeza(h).squeeze(-1)     # [B, T]: un valor por token

valor = FuncionDeValor().eval()
valor.load_state_dict(rm.state_dict())        # se inicializa desde el RM
ids = torch.tensor([[3, 7, 1, 1, 9, 4, 2, 5]])
with torch.no_grad():
    v = valor(ids)[0]
    print("V:", *[f"{x:+.2f}" for x in v[:4]])
    print("  ", *[f"{x:+.2f}" for x in v[4:]])
    print(f"RM de la respuesta: {rm(ids).item():+.2f}")
# <<<
