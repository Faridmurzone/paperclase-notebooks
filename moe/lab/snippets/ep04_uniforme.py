import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from _juguete import perdida_balanceo
T, E, k = 8, 4, 2
# >>>
# parejo: cada experto recibe 4 de las 16 asignaciones, P = 1/4
probs = torch.full((T, E), 1 / E)
idx = torch.tensor([[0, 1], [2, 3]] * 4)
print("parejo:    ", perdida_balanceo(probs, idx, E).item())
# colapsado: todo a los expertos 0 y 1, con prob. 0,45 cada uno
probs = torch.tensor([[0.45, 0.45, 0.05, 0.05]] * T)
idx = torch.tensor([[0, 1]] * T)
print("colapsado: ", round(perdida_balanceo(probs, idx, E).item(), 2))
# <<<
