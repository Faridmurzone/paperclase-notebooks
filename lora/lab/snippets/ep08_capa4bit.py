import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
N = torch.distributions.Normal(0, 1)
_neg = N.icdf(torch.linspace(0.03, 0.5, 8))
_pos = N.icdf(torch.linspace(0.5, 0.97, 9))
NF4 = torch.cat([_neg, _pos[1:]])
NF4 = NF4 / NF4.abs().max()


# >>>
class Lineal4bit(nn.Module):                  # base en 4 bits + LoRA
    def __init__(self, lin, r=8, alpha=16, bloque=64):
        super().__init__()
        w = lin.weight.detach().reshape(-1, bloque)
        absmax = w.abs().max(dim=1, keepdim=True).values
        codigo = ((w / absmax)[..., None] - NF4).abs().argmin(-1)
        self.register_buffer("codigo", codigo.to(torch.uint8))
        self.register_buffer("absmax", absmax)
        self.register_buffer("bias", lin.bias.detach().clone())
        self.forma = lin.weight.shape
        self.A = nn.Parameter(torch.randn(r, self.forma[1]) * 0.01)
        self.B = nn.Parameter(torch.zeros(self.forma[0], r))
        self.escala = alpha / r

    def forward(self, x):
        W = (NF4[self.codigo.long()] * self.absmax).reshape(self.forma)  # dequant
        lora = x @ self.A.T @ self.B.T            # el desvío, en float32
        return F.linear(x, W, self.bias) + self.escala * lora
# <<<


lin = nn.Linear(256, 256)
capa = Lineal4bit(lin)
x = torch.randn(4, 256)
print("codigo:", str(capa.codigo.dtype)[6:], tuple(capa.codigo.shape))
print("absmax:", tuple(capa.absmax.shape))
print("entrenables:", [n for n, p in capa.named_parameters()])
print(f"máx |lineal − 4 bits|: {(lin(x) - capa(x)).abs().max():.3f}")
