import torch
torch.manual_seed(0)
# >>>
for d in [2, 64, 2304]:
    V = torch.randn(1000, d)
    V = V / V.norm(dim=1, keepdim=True)       # 1000 direcciones unitarias
    C = V @ V.T                               # todos los cosenos de a pares
    pares = C[torch.triu(torch.ones(1000, 1000), 1).bool()]
    print(f"d={d:5}  |cos| promedio = {pares.abs().mean():.3f}"
          f"   máximo = {pares.abs().max():.3f}")
print("1/√2304 =", round(1 / 2304 ** 0.5, 3))
# <<<
