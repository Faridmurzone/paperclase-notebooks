import torch
import torch.nn.functional as F

torch.manual_seed(0)
# >>>
x = torch.randn(1, 3, 32, 32)                 # una imagen de 3 canales
w = torch.randn(16, 3, 3, 3)                  # 16 kernels de 3 × 3 × 3
cols = F.unfold(x, kernel_size=3, padding=1)  # (1, 27, 1024)
W = w.view(16, -1)                            # (16, 27)
y = (W @ cols).view(1, 16, 32, 32)            # una matmul y reacomodar
ref = F.conv2d(x, w, padding=1)
print("cols", tuple(cols.shape), " W", tuple(W.shape))
print("y   ", tuple(y.shape))
print("diferencia con conv2d:", f"{(y - ref).abs().max():.1e}")
# <<<
