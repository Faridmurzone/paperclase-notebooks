import math
import torch
torch.manual_seed(0)
# >>>
T, k, E, d, factor = 8, 1, 4, 3, 1.0
x = torch.randn(T, d)
exp = torch.tensor([0, 2, 0, 0, 1, 2, 0, 3])      # experto de cada token
C = math.ceil(factor * T * k / E)                  # = 2
orden = exp.argsort(stable=True)                   # agrupar, por llegada
cuenta = torch.bincount(exp, minlength=E)
pos = torch.arange(T) - (cuenta.cumsum(0) - cuenta)[exp[orden]]
entra = pos < C                                    # cabe en su fila
buffer = torch.zeros(E, C, d)                      # forma fija
quien = torch.full((E, C), -1)                     # -1 = lugar vacío
e, p, t = exp[orden][entra], pos[entra], orden[entra]
buffer[e, p], quien[e, p] = x[t], t
print("buffer:", tuple(buffer.shape), "| token en cada lugar:")
print(quien.tolist())
print("descartados:", orden[~entra].tolist())
# <<<
