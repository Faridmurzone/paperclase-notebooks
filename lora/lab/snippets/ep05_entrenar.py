import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import *
import _juguete
torch.manual_seed(0)
import json
red = agregar_lora(cargar_base(), r=4, alpha=8)
# >>>
W0 = red[0].base.weight.clone()                       # para comprobar después
loras = [p for p in red.parameters() if p.requires_grad]
opt = torch.optim.Adam(loras, lr=1e-2)                # solo A y B
g, historia = torch.Generator().manual_seed(0), []
for paso in range(301):
    i = torch.randint(0, len(Xtr), (64,), generator=g)
    perdida = F.cross_entropy(red(espejo(Xtr[i])), ytr[i])
    opt.zero_grad(); perdida.backward(); opt.step()
    historia.append(perdida.item())
    if paso % 100 == 0:
        print(f"paso {paso:3}: pérdida {perdida.item():.3f}")

print("en espejo:", f"{precision(red, espejo(Xte), yte):.1%}")
print("W0 intacta:", torch.equal(W0, red[0].base.weight))
print("estados de Adam:", len(opt.state), "tensores (A y B de 3 capas)")
# <<<
torch.save(red.state_dict(), os.path.join(DATOS, "ep05_lora.pt"))
json.dump([round(h, 4) for h in historia], open(os.path.join(DATOS, "ep05_historia.json"), "w"))
