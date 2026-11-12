import torch
import torch.nn.functional as F
# >>>
r = torch.tensor([0.9, 1.5, -0.4, 0.2])      # r(x, y) de las K = 4 respuestas
puesto = torch.tensor([2, 1, 3, 2])          # el ranking humano (A y D empatan)

dif = r[:, None] - r[None, :]                # dif[i, j] = r_i − r_j
gana = puesto[:, None] < puesto[None, :]     # i quedó mejor que j
perdidas = -F.logsigmoid(dif[gana])          # −log σ(r_w − r_l), por par
loss = perdidas.mean()                       # promedio: el 1 / C(K,2)

print("pares:", int(gana.sum()))
print("r_w − r_l:", [f"{v:+.1f}" for v in dif[gana]])
print("pérdida:  ", [f"{v:.2f}" for v in perdidas])
print(f"loss del prompt = {loss:.3f}")
# <<<
import json, os
json.dump({"dif": [[round(v, 1) for v in fila] for fila in dif.tolist()], "pares": [round(v, 1) for v in dif[gana].tolist()],
           "perdidas": [round(v, 2) for v in perdidas.tolist()], "loss": round(loss.item(), 3)},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep04_perdida.json"), "w"))
