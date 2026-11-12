import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import *
import _juguete
torch.manual_seed(0)
# >>>
capa = LoRALinear(nn.Linear(6, 5), r=2, alpha=4)
with torch.no_grad():
    capa.B.normal_()                                  # como después de entrenar
x = torch.randn(3, 6)                                 # 3 ejemplos
h = capa(x)
h.retain_grad()                                       # guardar δ = ∂L/∂h
perdida = (h ** 2).sum()
perdida.backward()

s, d = capa.escala, h.grad                            # d: 3 × 5
gB = s * d.T @ (x @ capa.A.T)                         # s · δ (Ax)ᵀ
gA = s * (d @ capa.B).T @ x                           # s · (Bᵀδ) xᵀ
print("∇B a mano = autograd:", torch.allclose(gB, capa.B.grad))
print("∇A a mano = autograd:", torch.allclose(gA, capa.A.grad))
print("∇W0:", capa.base.weight.grad)
# <<<
