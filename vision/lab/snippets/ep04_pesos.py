import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
# >>>
conv = nn.Conv2d(3, 4, kernel_size=3)          # 3 canales → 4 canales
W, b = conv.weight, conv.bias
print("peso:", tuple(W.shape), " bias:", tuple(b.shape))

x = torch.randn(1, 3, 8, 8)
y = conv(x)                                    # (1, 4, 6, 6)
# canal de salida 0 = suma sobre los 3 canales de entrada, + bias
a_mano = sum(F.conv2d(x[:, c:c+1], W[0:1, c:c+1]) for c in range(3)) + b[0]
print("salida:", tuple(y.shape))
print("diferencia:", f"{(y[:, 0:1] - a_mano).abs().max():.1e}")
# <<<
