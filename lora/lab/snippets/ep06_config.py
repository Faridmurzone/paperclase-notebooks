import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
torch.manual_seed(0)


class Bloque(nn.Module):                      # un bloque de juguete, d = 8
    def __init__(self, d=8):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)

    def forward(self, x):
        h = self.q_proj(x) + self.v_proj(x)
        return self.down(torch.relu(self.up(h)))


# >>>
cfg = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
                 target_modules=["q_proj", "v_proj"])
modelo = get_peft_model(Bloque(), cfg)
e, t = modelo.get_nb_trainable_parameters()
print(f"entrenables: {e} de {t} ({e / t:.1%})")

for nombre in ["q_proj", "v_proj", "up"]:
    mod = modelo.get_submodule("base_model.model." + nombre)
    origen = type(mod).__module__.split(".")[0]   # peft o torch
    print(f"{nombre:6} → {type(mod).__name__} de {origen}")

for n, p in modelo.named_parameters():
    if p.requires_grad:
        print("entrena:", n.removeprefix("base_model.model."))
# <<<
