import torch
import torch.nn.functional as F
# >>>
z = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0])
fs = {"sigm": torch.sigmoid, "tanh": torch.tanh,
      "ReLU": F.relu, "GELU": F.gelu}

fila = lambda t: " ".join(f"{round(float(v), 2) + 0:5.2f}" for v in t)
h = 1e-3                                  # pendiente numérica
print("z    ", fila(z))
for nombre, f in fs.items():
    pend = (f(z + h) - f(z - h)) / (2 * h)
    print(f"{nombre:4} ", fila(f(z)))     # valor
    print(" pend", fila(pend))            # pendiente
# <<<
