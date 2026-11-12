import torch
import torch.nn as nn
torch.manual_seed(0)
d, n_verdad, d_sae = 16, 48, 64
D = torch.randn(n_verdad, d); D /= D.norm(dim=1, keepdim=True)
def datos(B=2048):
    f = torch.rand(B, n_verdad) * (torch.rand(B, n_verdad) < 0.04)
    return f @ D
class SAE(nn.Module):
    def __init__(self, d, d_sae):
        super().__init__()
        self.W_e = nn.Parameter(torch.randn(d_sae, d) * 0.1)
        self.W_d = nn.Parameter(torch.randn(d, d_sae) * 0.1)
        self.b_e = nn.Parameter(torch.zeros(d_sae))
        self.b_d = nn.Parameter(torch.zeros(d))
    def forward(self, x):
        f = torch.relu((x - self.b_d) @ self.W_e.T + self.b_e)
        return f, f @ self.W_d.T + self.b_d
def perdida(x, f, x_hat, lam):
    return ((x - x_hat) ** 2).sum(-1).mean() + lam * f.abs().sum(-1).mean()
# >>>
def entrenar(lam, pasos=3000):
    torch.manual_seed(0)
    sae = SAE(d, d_sae)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    for _ in range(pasos):
        x = datos(); f, x_hat = sae(x)
        opt.zero_grad(); perdida(x, f, x_hat, lam).backward(); opt.step()
        with torch.no_grad():                      # columnas de W_d con norma 1
            sae.W_d /= sae.W_d.norm(dim=0, keepdim=True)
    return sae

for lam in (0.003, 0.3):
    sae = entrenar(lam)
    x = datos(); f, x_hat = sae(x)
    var = 1 - ((x - x_hat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
    rec = ((D @ sae.W_d).max(dim=1).values > 0.9).sum()   # ¿encontró las 48?
    print(f"λ={lam}: var. expl. {var:.2f}  L0 {(f > 0).sum(-1).float().mean():.1f}"
          f"  recuperadas {rec}/48")
# <<<
