import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import *
import _juguete
torch.manual_seed(0)
# >>>
def agregar_lora(red, r, alpha):
    red.requires_grad_(False)                         # congelar todo
    for i, capa in enumerate(red):
        if isinstance(capa, nn.Linear):               # cada capa lineal
            red[i] = LoRALinear(capa, r, alpha)       # se envuelve
    return red

red = agregar_lora(cargar_base(), r=4, alpha=8)       # la red del ep. 2
entrena = sum(p.numel() for p in red.parameters() if p.requires_grad)
total = sum(p.numel() for p in red.parameters())
print("capa 2:", type(red[2]).__name__,
      "A", tuple(red[2].A.shape), "B", tuple(red[2].B.shape))
print(f"entrenables: {entrena:,} de {total:,} ({entrena / total:.1%})")
print(f"en espejo, sin entrenar: {precision(red, espejo(Xte), yte):.1%}")
# <<<
