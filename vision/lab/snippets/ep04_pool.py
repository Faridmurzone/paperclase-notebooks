import torch
import torch.nn.functional as F

# >>>
x = torch.tensor([[1., 3., 0., 2.],
                  [4., 2., 1., 1.],
                  [0., 1., 5., 0.],
                  [2., 0., 3., 6.]])
y = F.max_pool2d(x[None, None], kernel_size=2)   # stride = 2 por defecto
print(y[0, 0])
print("forma:", tuple(x.shape), "→", tuple(y.shape[-2:]))
# <<<
