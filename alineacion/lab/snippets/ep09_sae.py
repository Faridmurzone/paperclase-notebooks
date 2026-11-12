import torch
import torch.nn as nn
torch.manual_seed(0)
d, n_verdad, d_sae = 16, 48, 64
D = torch.randn(n_verdad, d); D /= D.norm(dim=1, keepdim=True)   # 48 features en 16 dims
def datos(B=2048):
    f = torch.rand(B, n_verdad) * (torch.rand(B, n_verdad) < 0.04)  # ~2 activas
    return f @ D
# >>>
class SAE(nn.Module):
    def __init__(self, d, d_sae):
        super().__init__()
        self.W_e = nn.Parameter(torch.randn(d_sae, d) * 0.1)
        self.W_d = nn.Parameter(torch.randn(d, d_sae) * 0.1)
        self.b_e = nn.Parameter(torch.zeros(d_sae))
        self.b_d = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        f = torch.relu((x - self.b_d) @ self.W_e.T + self.b_e)   # features
        return f, f @ self.W_d.T + self.b_d                       # x̂

def perdida(x, f, x_hat, lam):
    return ((x - x_hat) ** 2).sum(-1).mean() + lam * f.abs().sum(-1).mean()

x = datos()
f, x_hat = SAE(d, d_sae)(x)
print("x", tuple(x.shape), "→ f", tuple(f.shape), "→ x̂", tuple(x_hat.shape))
print(f"pérdida al azar: {perdida(x, f, x_hat, 0.3).item():.2f}")
# <<<
