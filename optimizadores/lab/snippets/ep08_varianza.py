import torch
torch.manual_seed(0)

# >>>
n, b1, b2 = 100_000, 0.9, 0.999              # cien mil pesos, gradiente = ruido
m, v = torch.zeros(n), torch.zeros(n)
for t in range(1, 1001):
    g = torch.randn(n)                       # escala real: 1 en todos
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2
    mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
    if t in (1, 2, 10, 100, 1000):
        p99 = (1 / vh.sqrt()).quantile(0.99).item()   # 1/√v̂: el multiplicador
        paso = (mh / vh.sqrt()).abs().mean().item()  # |paso| / η
        print(f"t = {t:4}: 1/√v̂ p99 = {p99:5.1f}   |paso|/η = {paso:.2f}")
# <<<
