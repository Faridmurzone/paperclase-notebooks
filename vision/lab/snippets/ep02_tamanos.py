import torch
import torch.nn.functional as F

# >>>
x = torch.randn(1, 1, 32, 32)
w = torch.randn(1, 1, 3, 3)

def tam(n, k, p, s):
    return (n + 2 * p - k) // s + 1

for p, s in [(0, 1), (1, 1), (1, 2), (0, 2)]:
    y = F.conv2d(x, w, padding=p, stride=s)
    print(f"p={p} s={s}: fórmula {tam(32, 3, p, s):2d}"
          f"  torch {y.shape[-1]:2d}")
y = F.conv2d(x, w, padding="same")
print("padding='same':", tuple(y.shape[-2:]))
# <<<
