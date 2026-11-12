import torch
torch.manual_seed(0)
# >>>
p = torch.tensor([0.50, 0.25, 0.15, 0.10])   # 4 respuestas posibles
r = torch.tensor([1.0, 3.0, 0.0, -2.0])      # recompensa de cada una
print("suma de p:", p.sum().item())

esperanza = (p * r).sum()                    # sum_y p(y) r(y)
print("p * r:", [round(x, 2) for x in (p * r).tolist()])
print("E[r] exacta:", round(esperanza.item(), 3))

y = torch.multinomial(p, 10000, replacement=True)  # sortear
print("primeras 8 muestras:", y[:8].tolist())
print("sus r:", r[y[:8]].tolist())
print("promedio de 10000:", round(r[y].mean().item(), 3))
# <<<
