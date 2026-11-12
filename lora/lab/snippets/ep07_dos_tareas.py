import json
import os
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


base = Capa()
W0 = base.proj.weight.detach().clone()
X = torch.randn(256, d)
# dos tareas sintéticas: cada una pide un cambio de rango 2 distinto
tareas = {}
for nombre in ["tarea_a", "tarea_b"]:
    cambio = torch.randn(d, 2) @ torch.randn(2, d) * 0.3
    tareas[nombre] = X @ (W0 + cambio).T


def entrenar(modelo, Y, pasos=300):
    opt = torch.optim.Adam([p for p in modelo.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(pasos):
        perdida = ((modelo(X) - Y) ** 2).mean()
        opt.zero_grad()
        perdida.backward()
        opt.step()


def error(modelo, Y):
    with torch.no_grad():
        return ((modelo(X) - Y) ** 2).mean().item()


# >>>
cfg = LoraConfig(r=2, lora_alpha=4, target_modules=["proj"])
modelo = get_peft_model(base, cfg, adapter_name="tarea_a")
modelo.add_adapter("tarea_b", cfg)               # segundo adapter

for nombre, Y in tareas.items():
    modelo.set_adapter(nombre)                 # activo y entrenable
    entrenar(modelo, Y)

filas = {}
for activo in ["tarea_a", "tarea_b", None]:
    modelo.set_adapter(activo or "tarea_a")
    if activo is None:                          # base sola
        with modelo.disable_adapter():
            filas["base"] = [error(modelo, Y) for Y in tareas.values()]
    else:
        filas[activo] = [error(modelo, Y) for Y in tareas.values()]
print("error en:  tarea_a tarea_b")
for k, v in filas.items():
    print(f"{k:9} {v[0]:7.3f} {v[1]:7.3f}")
# <<<
json.dump(filas, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep07_tareas.json"), "w"))
