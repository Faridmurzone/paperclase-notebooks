import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from _juguete import Experto
torch.manual_seed(0)
x = torch.tensor([1.0, 0.5, -1.0])
W_r = torch.tensor([[1.0, 0.0, 0.5, -1.0], [0.0, 2.0, 0.0, 1.0], [0.5, 0.0, -1.0, 0.0]])
# >>>
expertos = [Experto(d=3, h=4) for _ in range(4)]
probs = (x @ W_r).softmax(-1)
top = probs.topk(2)
g = top.values / top.values.sum()
# solo los elegidos: 2 expertos calculados
y = sum(gi * expertos[i](x) for gi, i in zip(g, top.indices.tolist()))
# los 4, con g = 0 en los no elegidos
g_todos = torch.zeros(4).scatter(0, top.indices, g)
y_denso = sum(g_todos[i] * expertos[i](x) for i in range(4))
print("g completo:", [round(v, 2) for v in g_todos.tolist()])
print("y:      ", [round(v, 3) for v in y.tolist()])
print("y_denso:", [round(v, 3) for v in y_denso.tolist()])
# <<<
