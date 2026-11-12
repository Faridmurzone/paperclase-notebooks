import torch
from torch import nn
X = torch.tensor([[1.0, 4.0, 10.0, 0.0], [2.0, 6.0, 30.0, 1.0], [3.0, 2.0, 20.0, -1.0], [6.0, 8.0, 40.0, 0.0]])
torch.set_printoptions(precision=3, sci_mode=False)
# >>>
bn = nn.BatchNorm1d(4)                # momentum=0.1 por defecto
bn(X)                                 # un forward en modo train
print("running_mean:", bn.running_mean.numpy().round(3))
print("0,1 · μ_B   :", (0.1 * X.mean(0)).numpy().round(3))
print("running_var[0]:", round(bn.running_var[0].item(), 3))
print("var no sesgada:", round(X[:, 0].var().item(), 3))

bn.eval()                             # en eval usa las running
print("eval, fila 0:", bn(X)[0].detach().numpy().round(2))

uno = X[:1]                           # un batch de un solo ejemplo
bn.train()
try:
    bn(uno)
except ValueError:
    print("BatchNorm: ValueError")
print("LayerNorm:", nn.LayerNorm(4)(uno)[0].detach().numpy().round(2))
# <<<
