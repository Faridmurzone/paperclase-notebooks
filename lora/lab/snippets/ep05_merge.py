import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import *
import _juguete
torch.manual_seed(0)
red = agregar_lora(cargar_base(), r=4, alpha=8)
red.load_state_dict(torch.load(os.path.join(DATOS, "ep05_lora.pt")))
# >>>
def fusionar(capa):                                   # LoRALinear → nn.Linear
    base = capa.base
    nueva = nn.Linear(base.in_features, base.out_features)
    with torch.no_grad():
        dW = capa.escala * capa.B @ capa.A            # d × k, se arma una vez
        nueva.weight.copy_(capa.base.weight + dW)     # W = W0 + (α/r)·BA
        nueva.bias.copy_(capa.base.bias)
    return nueva, dW

fusionada = nn.Sequential(*[fusionar(c)[0] if isinstance(c, LoRALinear)
                            else c for c in red])
X = espejo(Xte)
with torch.no_grad():
    dif = (red(X) - fusionada(X)).abs().max().item()
print(f"diferencia máxima de logits: {dif:.1e}")
p1, p2 = precision(red, X, yte), precision(fusionada, X, yte)
print(f"precisión: {p1:.1%} → {p2:.1%}")
dW0 = fusionar(red[0])[1]
print("rango de ΔW, capa 0:", torch.linalg.matrix_rank(dW0).item())
print("parámetros:", sum(p.numel() for p in fusionada.parameters()))
# <<<
