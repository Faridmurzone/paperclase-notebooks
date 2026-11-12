import torch
torch.manual_seed(0)
# >>>
for d in [4, 64, 256, 2304]:
    q = torch.randn(10_000, d)     # 10.000 pares de vectores al azar
    k = torch.randn(10_000, d)     # coordenadas de varianza 1
    punto = (q * k).sum(dim=1)     # q·k para cada par
    print(f"d={d:5}  desvío de q·k = {punto.std():6.2f}"
          f"   √d = {d ** 0.5:5.1f}"
          f"   desvío de q·k/√d = {(punto / d ** 0.5).std():.2f}")
# <<<
