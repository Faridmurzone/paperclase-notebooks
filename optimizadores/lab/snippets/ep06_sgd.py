import torch

def grad_perdida(w):                            # pérdida de juguete: 0,5·‖w − (3, −2)‖²
    return w - torch.tensor([3.0, -2.0])

# >>>
eta, lam = 0.1, 0.5
a = torch.tensor([1.0, 1.0])                    # A: L2 dentro del gradiente
b = torch.tensor([1.0, 1.0])                    # B: achicar aparte
c = torch.tensor([1.0, 1.0], requires_grad=True)
opt = torch.optim.SGD([c], lr=eta, weight_decay=lam)
for t in range(10):
    a = a - eta * (grad_perdida(a) + lam * a)
    b = (1 - eta * lam) * b - eta * grad_perdida(b)
    c.grad = grad_perdida(c.detach()); opt.step()
print("A, L2      :", a.numpy().round(5))
print("B, decay   :", b.numpy().round(5))
print("C, torch   :", c.detach().numpy().round(5))
print(f"factor 1 − ηλ = {1 - eta * lam}")
print(f"máx |A − B| = {(a - b).abs().max():.1e}")
# <<<
