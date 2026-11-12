import torch
torch.manual_seed(0)

# >>>
G = torch.tensor([[1.0, 2.0, 0.5, 1.0],
                  [2.0, 4.0, 1.0, 2.0],
                  [0.5, 1.0, 3.0, 0.5]])           # gradiente de una matriz 3 × 4
V = G ** 2                                         # lo que guardaría Adam: 12 números
R = V.sum(dim=1)                                   # suma de cada fila: 3 números
C = V.sum(dim=0)                                   # suma de cada columna: 4 números
V_hat = torch.outer(R, C) / R.sum()                # Adafactor: rango 1
print("R:", R.tolist(), " C:", C.tolist())
print("V real    fila 1:", V[0].tolist())
print("V factor. fila 1:", [round(x, 2) for x in V_hat[0].tolist()])
print("V real    fila 3:", V[2].tolist())
print("V factor. fila 3:", [round(x, 2) for x in V_hat[2].tolist()])
print(f"números guardados: {V.numel()} → {R.numel() + C.numel()}")
# <<<
