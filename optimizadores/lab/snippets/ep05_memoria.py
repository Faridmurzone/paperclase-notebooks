import torch
from torch import nn
torch.manual_seed(0)

# >>>
modelo = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 10))
n = sum(p.numel() for p in modelo.parameters())
for nombre, opt in [("SGD", torch.optim.SGD(modelo.parameters(), lr=0.1)),
                    ("momentum", torch.optim.SGD(modelo.parameters(), lr=0.1,
                                                 momentum=0.9)),
                    ("Adam", torch.optim.Adam(modelo.parameters()))]:
    modelo(torch.randn(4, 64)).sum().backward()
    opt.step()                                  # crea el estado
    estado = [v for s in opt.state.values() for v in s.values()
              if torch.is_tensor(v) and v.numel() > 1]
    extra = sum(v.numel() for v in estado)
    print(f"{nombre:8}: {extra:6} números extra ({extra / n:.0f}× los pesos)")
print(f"pesos:    {n:6}")
# <<<
