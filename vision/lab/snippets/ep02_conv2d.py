import torch
import torch.nn.functional as F

torch.manual_seed(0)
# >>>
x = torch.randn(1, 1, 8, 8)          # (batch, canales, alto, ancho)
w = torch.randn(1, 1, 3, 3)          # (salidas, entradas, kh, kw)

a_mano = torch.zeros(6, 6)
for i in range(6):
    for j in range(6):
        a_mano[i, j] = (x[0, 0, i:i+3, j:j+3] * w[0, 0]).sum()

con_torch = F.conv2d(x, w)           # el mismo deslizamiento, en C
print("forma:", tuple(con_torch.shape))
print(f"diferencia máxima: {(con_torch[0, 0] - a_mano).abs().max():.1e}")
capa = torch.nn.Conv2d(1, 1, kernel_size=3)
print("weight:", tuple(capa.weight.shape), " bias:", tuple(capa.bias.shape))
# <<<
