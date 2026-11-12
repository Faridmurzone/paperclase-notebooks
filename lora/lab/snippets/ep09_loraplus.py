import json
import os
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
d = 64
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")


class Capa(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.proj(x)


def tarea(tipo="azar"):
    """Base al azar y una tarea sintética: un cambio al azar de rango completo, o reescalar las filas de W."""
    torch.manual_seed(0)
    base = Capa()
    W0 = base.proj.weight.detach()
    g = torch.Generator().manual_seed(123)
    cambio = torch.randn(d, d, generator=g) * 0.1
    X = torch.randn(512, d, generator=g)
    if tipo == "azar":
        Y = X @ (W0 + cambio).T
    else:                                       # cada fila de W, estirada o encogida
        escala = torch.exp(0.3 * torch.randn(d, 1, generator=g))   # siempre > 0
        Y = X @ (escala * W0).T
    return base, X, Y


def entrenar(r=16, rs=False, dora=False, lam=1.0, lr=1e-3, pasos=150, tipo="azar"):
    base, X, Y = tarea(tipo)
    cfg = LoraConfig(r=r, lora_alpha=8, target_modules=["proj"], use_rslora=rs, use_dora=dora)
    m = get_peft_model(base, cfg)
    A = [p for n, p in m.named_parameters() if p.requires_grad and "lora_B" not in n]
    B = [p for n, p in m.named_parameters() if "lora_B" in n]
    opt = torch.optim.Adam([{"params": A, "lr": lr}, {"params": B, "lr": lam * lr}])
    hist = []
    for _ in range(pasos):
        perdida = ((m(X) - Y) ** 2).mean()
        opt.zero_grad()
        perdida.backward()
        opt.step()
        hist.append(perdida.item())
    return hist


# >>>
# LoRA+: B aprende λ veces más rápido que A
def grupos(modelo, lr, lam):
    A = [p for n, p in modelo.named_parameters() if "lora_A" in n]
    B = [p for n, p in modelo.named_parameters() if "lora_B" in n]
    return [{"params": A, "lr": lr}, {"params": B, "lr": lam * lr}]

curvas = {}
for lam in [1, 4, 16]:                       # λ = 1 es LoRA común
    curvas[lam] = entrenar(r=16, lam=lam)
    h = curvas[lam]
    print(f"λ = {lam:2}: paso 25 {h[25]:.3f}, paso 50 {h[50]:.3f},",
          f"final {h[-1]:.3f}")
# <<<
json.dump(curvas, open(os.path.join(DATOS, "ep09_loraplus.json"), "w"))
