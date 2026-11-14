import torch
torch.manual_seed(0)
x = torch.randn(64, 16, 8, 8) * 3 + 5        # batch de 64, 16 canales, 8×8; media 5, desvío 3
# >>>
def batchnorm(x, gamma, beta, eps=1e-5):
    mu = x.mean(dim=(0, 2, 3), keepdim=True)            # μ_B: una por canal
    var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
    x_hat = (x - mu) / torch.sqrt(var + eps)             # media 0, var 1
    return gamma.view(1, -1, 1, 1) * x_hat + beta.view(1, -1, 1, 1)

bn = torch.nn.BatchNorm2d(16)                # γ = 1, β = 0 al inicio
y = batchnorm(x, bn.weight, bn.bias)
r = lambda t: f"{abs(t.mean()):.2f} / {t.std():.2f}"
print("media / desvío antes:  ", r(x))
print("media / desvío después:", r(y))
print("dif. con nn.BatchNorm2d:", f"{(y - bn(x)).abs().max():.1e}")
print("valores por canal:", 64 * 8 * 8)
# <<<
