import json
import os
import tempfile
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file


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


def base():                                   # siempre la misma base
    torch.manual_seed(0)
    return Bloque()


cfg = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["q_proj", "v_proj"])
modelo = get_peft_model(base(), cfg)
with torch.no_grad():
    for n, p in modelo.named_parameters():
        if "lora_B" in n:
            p.normal_()
carpeta = tempfile.mkdtemp()
x = torch.randn(3, 8)
# >>>
modelo.save_pretrained(carpeta)
for f in sorted(os.listdir(carpeta)):
    print(f, os.path.getsize(os.path.join(carpeta, f)), "bytes")

pesos = load_file(os.path.join(carpeta, "adapter_model.safetensors"))
for k, v in pesos.items():
    print(" ", k.removeprefix("base_model.model."), tuple(v.shape))
c = json.load(open(os.path.join(carpeta, "adapter_config.json")))
print("config:", c["r"], c["lora_alpha"], sorted(c["target_modules"]))

otro = PeftModel.from_pretrained(base(), carpeta)   # base + adapter
print("misma salida:", torch.allclose(modelo(x), otro(x)))
# <<<
