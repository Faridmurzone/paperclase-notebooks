import torch

# >>>
g = torch.tensor([0.5, -4.0])                 # el mismo gradiente en cada paso
for Clase in [torch.optim.Adam, torch.optim.RAdam]:
    w = torch.zeros(2, requires_grad=True)
    opt = Clase([w], lr=0.1)
    pasos = []
    for t in range(1, 501):
        antes = w.detach().clone()
        w.grad = g.clone(); opt.step()
        pasos.append((w.detach() - antes).tolist())
    print(Clase.__name__)
    for t in (1, 5, 6, 50, 500):
        print(f"  paso {t}: {[round(x, 3) for x in pasos[t - 1]]}")
# <<<
