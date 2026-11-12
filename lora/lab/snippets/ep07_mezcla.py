import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
torch.manual_seed(0)
d = 16


class Capa(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.proj(x)


cfg = LoraConfig(r=2, lora_alpha=4, target_modules=["proj"])
modelo = get_peft_model(Capa(), cfg, adapter_name="tarea_a")
modelo.add_adapter("tarea_b", cfg)
with torch.no_grad():
    for n, p in modelo.named_parameters():
        if "lora_B" in n:
            p.normal_()
x = torch.randn(4, d)                          # 4 pedidos distintos
# >>>
modelo.set_adapter("tarea_b")
for n, p in modelo.named_parameters():
    if p.requires_grad:
        print("entrena:", n.removeprefix("base_model.model."))

modelo.eval()
quien = ["tarea_a", "tarea_b", "__base__", "tarea_a"]  # uno por fila
mezcla = modelo(x, adapter_names=quien)

for i, nombre in enumerate(quien):
    if nombre == "__base__":
        with modelo.disable_adapter():
            solo = modelo(x)
    else:
        modelo.set_adapter(nombre)
        solo = modelo(x)
    print(f"fila {i} ({nombre}): diferencia",
          (mezcla[i] - solo[i]).abs().max().item())
# <<<
