import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import *
torch.manual_seed(0)
# >>>
red = nn.Sequential(nn.Linear(64, 256), nn.ReLU(),     # 8×8 píxeles
                    nn.Linear(256, 256), nn.ReLU(),    # la capa del medio
                    nn.Linear(256, 10))                # un logit por dígito
print(f"parámetros: {sum(p.numel() for p in red.parameters()):,}")

entrenar(red, Xtr, ytr, pasos=1500)                    # "preentrenamiento"
print("dígitos normales: ", f"{precision(red, Xte, yte):.1%}")
print("dígitos en espejo:", f"{precision(red, espejo(Xte), yte):.1%}")
# <<<
torch.save(red.state_dict(), os.path.join(DATOS, "ep02_base.pt"))
