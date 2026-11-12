import math
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
torch.manual_seed(0)


class Bloque(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)

    def forward(self, x):
        h = self.q_proj(x) + self.v_proj(x)
        return self.down(torch.relu(self.up(h)))


cfg = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["q_proj", "v_proj"])
modelo = get_peft_model(Bloque(), cfg)
# >>>
q = modelo.base_model.model.q_proj           # el lora.Linear
print("base_layer:", tuple(q.base_layer.weight.shape))
A = q.lora_A["default"].weight               # r × k
B = q.lora_B["default"].weight               # d × r
print("lora_A:", tuple(A.shape), " lora_B:", tuple(B.shape))
print("scaling:", q.scaling["default"])      # lora_alpha / r

print("B al empezar, máx |B|:", B.abs().max().item())
k = A.shape[1]
print(f"A en [{A.min():.2f}, {A.max():.2f}],",
      f"cota 1/√k = {1 / math.sqrt(k):.2f}")
# <<<
