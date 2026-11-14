import torch
torch.manual_seed(0)
x = torch.randn(64, 3, 8, 8)
objetivo = torch.randn(64, 16, 8, 8)               # una pérdida cualquiera, fija
# >>>
conv = torch.nn.Conv2d(3, 16, 3, padding=1, bias=False)
bn = torch.nn.BatchNorm2d(16)                # modo train: media del batch

def paso(escala):
    w = (conv.weight.detach() * escala).requires_grad_(True)
    z = torch.nn.functional.conv2d(x, w, padding=1)
    y = bn(z)
    ((y - objetivo) ** 2).mean().backward()
    return z.std().item(), y, w.grad.norm().item()

s1, y1, g1 = paso(1.0)
s10, y10, g10 = paso(10.0)                   # pesos 10 veces más grandes
print("desvío antes de la BN:", round(s1, 2), "→", round(s10, 2))
print("después de la BN, dif. máx.:", f"{(y10 - y1).abs().max():.1e}")
print("gradiente con pesos ×10 / ×1:", round(g10 / g1, 3))
# <<<
