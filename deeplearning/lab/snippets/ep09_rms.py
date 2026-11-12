import torch
from torch import nn
torch.set_printoptions(precision=2, sci_mode=False)
# >>>
x = torch.tensor([1.0, 4.0, 10.0, 1.0])     # una fila, 4 features

ln = nn.LayerNorm(4, elementwise_affine=False)
rms = nn.RMSNorm(4, elementwise_affine=False)
print("LayerNorm:", ln(x).numpy().round(2))
print("RMSNorm  :", rms(x).numpy().round(2))

raiz = x.pow(2).mean().sqrt()                # raíz cuadrática media
print("x / raíz :", (x / raiz).numpy().round(2))
print("raíz =", round(raiz.item(), 2))
print("medias:", round(ln(x).mean().item(), 2), round(rms(x).mean().item(), 2))
# <<<
