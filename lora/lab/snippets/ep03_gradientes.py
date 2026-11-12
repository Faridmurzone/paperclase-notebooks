import torch
torch.manual_seed(0)
d, k, r = 6, 4, 2
W0 = torch.randn(d, k)
x, objetivo = torch.randn(8, k), torch.randn(8, d)
# >>>
def probar(nombre, A, B, pasos=10):
    inicio = (B @ A).norm().item()                   # ¿arranca cambiado?
    A.requires_grad_(); B.requires_grad_()
    opt = torch.optim.Adam([A, B], lr=0.01)
    for paso in range(pasos):
        h = x @ W0.T + (x @ A.T) @ B.T               # forward con LoRA
        perdida = ((h - objetivo) ** 2).mean()
        opt.zero_grad(); perdida.backward()
        if paso == 0:
            gA, gB = A.grad.norm().item(), B.grad.norm().item()
        opt.step()
    fin = (B @ A).norm().item()
    print(f"{nombre}: |BA| {inicio:.2f} → {fin:.2f}"
          f" | paso 1: gA={gA:.2f} gB={gB:.2f}")

probar("A azar, B cero", torch.randn(r, k), torch.zeros(d, r))
probar("A cero, B cero", torch.zeros(r, k), torch.zeros(d, r))
probar("A azar, B azar", torch.randn(r, k), torch.randn(d, r))
# <<<
