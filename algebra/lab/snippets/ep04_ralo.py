import torch
torch.manual_seed(0)
# >>>
a = torch.tensor([3.0, 0.4, -0.2, 1.5, 0.1])   # lo que «querría» valer w
lam = 1.0

def ajustar(castigo):
    w = torch.zeros(5, requires_grad=True)
    opt = torch.optim.SGD([w], lr=0.002)
    for _ in range(5000):
        perdida = ((w - a) ** 2).sum() + lam * castigo(w)
        opt.zero_grad(); perdida.backward(); opt.step()
    return w.detach()

w2 = ajustar(lambda w: (w ** 2).sum())     # castigo L2²
w1 = ajustar(lambda w: w.abs().sum())      # castigo L1
print("con L2²:", w2.numpy().round(2))
print("con L1 :", w1.numpy().round(2))
ceros = lambda w: int((w.abs() < 0.01).sum())
print("ceros:", ceros(w2), "con L2²,", ceros(w1), "con L1")
# <<<
