import torch
import torch.nn.functional as F
from torch import nn
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16, dtype=torch.float32); y = torch.tensor(y)

# >>>
torch.manual_seed(0)
modelo = nn.Linear(64, 10)
b = torch.arange(8)                                 # un minibatch de 8

perdida = F.cross_entropy(modelo(X[b]), y[b])       # promedio de 8
perdida.backward()
g_batch = modelo.weight.grad.clone()

suma = torch.zeros_like(g_batch)
for i in b:                                         # uno por vez
    modelo.zero_grad()
    F.cross_entropy(modelo(X[i:i + 1]), y[i:i + 1]).backward()
    suma += modelo.weight.grad
dif = (g_batch - suma / 8).abs().max().item()
print(f"forma del gradiente: {tuple(g_batch.shape)}")
print(f"diferencia máxima: {dif:.1e}")
# <<<
