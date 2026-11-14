import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
# >>>
class BloqueResidual(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.atajo = nn.Identity()               # el camino directo
        if stride != 1 or c_in != c_out:         # proyección 1×1
            self.atajo = nn.Conv2d(c_in, c_out, 1, stride, bias=False)

    def forward(self, x):
        f = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return F.relu(f + self.atajo(x))         # y = relu(F(x) + x)

x = torch.randn(8, 16, 32, 32)
print("igual:  ", tuple(BloqueResidual(16, 16)(x).shape))
print("achica: ", tuple(BloqueResidual(16, 32, stride=2)(x).shape))
# <<<
