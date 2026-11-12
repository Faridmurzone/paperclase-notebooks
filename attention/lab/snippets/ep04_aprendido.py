import torch
import torch.nn as nn
torch.manual_seed(0)
d = 512
def sinusoidal(pos, d):
    omega = 1 / 10000 ** (torch.arange(0, d, 2) / d)
    pe = torch.zeros(d)
    pe[0::2], pe[1::2] = torch.sin(pos * omega), torch.cos(pos * omega)
    return pe
# >>>
aprendido = nn.Embedding(512, d)   # una fila entrenable por posición
print("parámetros extra:", aprendido.weight.numel())

for pos in [100, 600]:             # entrenado con frases de ≤ 512
    try:
        v = aprendido(torch.tensor(pos))
        print(f"aprendido,  pos {pos}: forma {tuple(v.shape)}")
    except IndexError:
        print(f"aprendido,  pos {pos}: IndexError, no hay fila")
    print(f"sinusoidal, pos {pos}: forma {tuple(sinusoidal(pos, d).shape)}")
# <<<
