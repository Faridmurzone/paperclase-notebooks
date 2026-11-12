import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import *
import _juguete
torch.manual_seed(0)
_juguete_LoRALinear = LoRALinear
# >>>
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)          # W0 y b: congelados
        k, d = base.in_features, base.out_features
        self.A = nn.Parameter(torch.randn(r, k) / k**0.5)  # r × k, al azar
        self.B = nn.Parameter(torch.zeros(d, r))        # d × r, en cero
        self.escala = alpha / r

    def forward(self, x):
        return self.base(x) + self.escala * (x @ self.A.T @ self.B.T)

capa = LoRALinear(nn.Linear(64, 256), r=4, alpha=8)
for nombre, p in capa.named_parameters():
    print(f"{nombre:12} {str(tuple(p.shape)):10} entrena={p.requires_grad}")
x = torch.randn(5, 64)
print("igual a la base:", torch.equal(capa(x), capa.base(x)))
# <<<
import inspect
assert inspect.getsource(LoRALinear).split("\n", 1)[1] == inspect.getsource(_juguete_LoRALinear).split('"""', 2)[2].lstrip("\n"), "la clase del video y la de _juguete difieren"
