import torch
lp_old = torch.zeros(6)                               # log π_old de seis tokens
lp = torch.log(torch.tensor([0.5, 1.0, 1.5, 0.5, 1.0, 1.5])).requires_grad_(True)
A = torch.tensor([1., 1., 1., -1., -1., -1.])         # ventaja de cada token
# >>>
def perdida_ppo(lp, lp_old, A, eps=0.2):
    ratio = (lp - lp_old).exp()                       # π / π_old
    s1 = ratio * A
    s2 = ratio.clamp(1 - eps, 1 + eps) * A            # recortado a [0.8, 1.2]
    return -torch.min(s1, s2).mean(), ratio

loss, ratio = perdida_ppo(lp, lp_old, A)
loss.backward()
print("ratio:   ", ratio.detach().numpy().round(2))
print("ventaja: ", A.numpy())
print("¿empuja?:", (lp.grad != 0).numpy())
# <<<
