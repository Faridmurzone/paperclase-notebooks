import json, os
import numpy as np
import torch
from sklearn.datasets import load_digits
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
# >>>
X, y = load_digits(return_X_y=True)        # viene con scikit-learn
print("X:", X.shape, " y:", y.shape)       # 1797 imágenes de 8x8
print("píxeles de", X.min(), "a", X.max())
print("clases:", np.unique(y))

X = torch.tensor(X / 16, dtype=torch.float32)   # píxeles en [0, 1]
y = torch.tensor(y)
g = torch.Generator().manual_seed(0)            # semilla: reproducible
perm = torch.randperm(len(X), generator=g)      # mezclar los índices
train, val = perm[:1200], perm[1200:]
print("train:", len(train), " val:", len(val))
n = torch.bincount(y[val])            # ejemplos de cada dígito
print("por clase en val:", n.min().item(), "a", n.max().item())
# <<<
os.makedirs(DATOS, exist_ok=True)
i = int(train[0])
json.dump({"digito": (X[i] * 16).reshape(8, 8).int().tolist(), "etiqueta": int(y[i])},
          open(os.path.join(DATOS, "ep07_digito.json"), "w"))
