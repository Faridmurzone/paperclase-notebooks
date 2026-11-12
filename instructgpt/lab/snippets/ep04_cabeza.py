import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.utils import logging; logging.set_verbosity_error()
torch.manual_seed(0)
cfg = GPT2Config(vocab_size=16, n_positions=16, n_embd=64, n_layer=2, n_head=4, bos_token_id=1, eos_token_id=1)
sft = GPT2LMHeadModel(cfg).eval()                   # hace de modelo SFT
# >>>
print("unembedding del SFT:", tuple(sft.lm_head.weight.shape))  # [V, d]

class RewardModel(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.cuerpo = lm.transformer                # todo menos el unembedding
        self.cabeza = nn.Linear(lm.config.n_embd, 1)  # proyección a un número
    def forward(self, ids):
        h = self.cuerpo(input_ids=ids).last_hidden_state   # [B, T, d]
        return self.cabeza(h[:, -1]).squeeze(-1)    # último token → r(x, y)

rm = RewardModel(sft)
ids = torch.tensor([[5, 9, 3, 3, 12, 7, 4]])        # prompt + respuesta
print("logits del SFT:", tuple(sft(input_ids=ids).logits.shape))
print("salida del RM: ", tuple(rm(ids).shape))
print(f"r(x, y) = {rm(ids).item():+.3f}")
# <<<
