import math
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.datasets import load_digits
torch.manual_seed(0)
# >>>
d = load_digits()
X = torch.tensor(d.data[:256], dtype=torch.float32) / 16   # píxeles 0..1
y = torch.tensor(d.target[:256])                            # dígito 0..9

red = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
logits = red(X)                       # (256, 10), sin softmax
L = F.cross_entropy(logits, y)        # softmax + -log, adentro
print("sin entrenar:     ", round(L.item(), 3))
print("azar, -log(1/10): ", round(math.log(10), 3))
# <<<
