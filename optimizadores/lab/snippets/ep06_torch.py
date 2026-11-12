import torch

def ruido(t):
    return torch.tensor([10.0, 0.01]) * (1 if t % 2 == 0 else -1)

# >>>
def correr(Clase):
    th = torch.tensor([1.0, 1.0], requires_grad=True)
    opt = Clase([th], lr=0.01, weight_decay=0.1)
    for t in range(1, 301):
        th.grad = ruido(t)                     # solo el gradiente de la pérdida
        opt.step()
    return th.detach().numpy().round(3)

print("torch.optim.Adam(weight_decay=0.1): ", correr(torch.optim.Adam))
print("torch.optim.AdamW(weight_decay=0.1):", correr(torch.optim.AdamW))
# <<<
