import torch
import torch.nn as nn
torch.manual_seed(0)
capa = nn.Linear(4, 4)
x = torch.randn(1, 4)
# >>>
def mirar(module, input, output):        # se llama DESPUÉS del forward
    print(type(module).__name__, type(input).__name__, len(input))
    # devolver None = no tocar nada

def anular(module, input, output):
    return output * 0                    # lo que devuelve REEMPLAZA la salida

def duplicar_entrada(module, args):      # pre-hook: ANTES del forward
    return (args[0] * 2,)                # reemplaza los argumentos

h1 = capa.register_forward_hook(mirar)
print(capa(x).abs().sum().item() > 0)
h2 = capa.register_forward_hook(anular)
print(capa(x).sum().item())
h1.remove(); h2.remove()                 # sin remove, el hook queda para siempre
h3 = capa.register_forward_pre_hook(duplicar_entrada)
print(torch.allclose(capa(x), capa.weight @ (2 * x[0]) + capa.bias))
# <<<
