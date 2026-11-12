import json, os
import torch
from torch import nn
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
torch.manual_seed(0)
# >>>
drop = nn.Dropout(p=0.5)        # apaga cada neurona con prob. 0,5
h = torch.tensor([0.5, 1.0, 1.5, 2.0, 0.8, 1.2])   # 6 activaciones

drop.train()                    # modo entrenamiento
print("h      :", h.numpy())
print("train 1:", drop(h).numpy().round(2))
print("train 2:", drop(h).numpy().round(2))   # otra máscara al azar

drop.eval()                     # modo evaluación
print("eval   :", drop(h).numpy().round(2))

drop.train()                    # promedio de 10.000 máscaras
media = torch.stack([drop(h) for _ in range(10_000)]).mean(0)
print("media  :", media.numpy().round(2))
# <<<
torch.manual_seed(1)
json.dump([[round(v, 2) for v in drop(h).tolist()] for _ in range(4)], open(os.path.join(DATOS, "ep08_mascaras.json"), "w"))
