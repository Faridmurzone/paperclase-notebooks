import torch
# >>>
w = torch.tensor([1.0, -2.0], requires_grad=True)
opt = torch.optim.SGD([w], lr=0.1, weight_decay=0.5)   # η = 0,1 y λ = 0,5

loss = (w * 0).sum()          # una pérdida que no empuja: g = 0
loss.backward()
opt.step()                    # solo actúa el weight decay
print("w después de un paso:", w.detach().numpy().round(4))
print("w · (1 − η·λ)       :", (torch.tensor([1.0, -2.0]) * (1 - 0.1 * 0.5)).numpy())

for paso in range(9):         # nueve pasos más, sin gradiente
    opt.zero_grad(); (w * 0).sum().backward(); opt.step()
print("después de 10 pasos :", w.detach().numpy().round(4))
print("0,95 ** 10          :", round(0.95 ** 10, 4))
# <<<
