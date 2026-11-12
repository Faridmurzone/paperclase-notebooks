import torch
from torch import nn
X = torch.tensor([[1.0, 4.0, 10.0, 0.0], [2.0, 6.0, 30.0, 1.0], [3.0, 2.0, 20.0, -1.0], [6.0, 8.0, 40.0, 0.0]])
bn_mano = (X - X.mean(0)) / torch.sqrt(X.var(0, unbiased=False) + 1e-5)
ln_mano = (X - X.mean(1, keepdim=True)) / torch.sqrt(X.var(1, unbiased=False, keepdim=True) + 1e-5)
# >>>
bn = nn.BatchNorm1d(4)      # 4 features: un γ y un β por columna
ln = nn.LayerNorm(4)        # filas de 4: un γ y un β por posición
print("γ:", bn.weight.tolist())
print("β:", bn.bias.tolist())

print("BatchNorm:", torch.allclose(bn(X), bn_mano, atol=1e-5))
print("LayerNorm:", torch.allclose(ln(X), ln_mano, atol=1e-5))

with torch.no_grad():
    bn.weight.fill_(2.0); bn.bias.fill_(0.5)   # γ = 2, β = 0,5
print("γ=2, β=0,5:", bn(X)[:, 0].detach().numpy().round(2))
print("2·x̂ + 0,5:", (2 * bn_mano[:, 0] + 0.5).numpy().round(2))
# <<<
