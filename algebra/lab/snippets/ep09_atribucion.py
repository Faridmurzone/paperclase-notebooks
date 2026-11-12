import torch
torch.manual_seed(0)
r = torch.randn(4)
# >>>
def metrica(a):                           # una métrica no lineal de a
    return torch.tanh(a) @ r

a_corr = torch.tensor([0.2, -0.1, 0.4, 0.0], requires_grad=True)
m = metrica(a_corr)
m.backward()                              # un solo backward
grad = a_corr.grad
d = torch.tensor([0.5, -0.5, 0.5, 0.5])   # dirección, largo 1

for escala in [0.05, 0.3, 1.5]:
    delta = escala * d                    # a_limpio - a_corr
    real = metrica(a_corr.detach() + delta) - m.detach()
    lineal = delta @ grad                 # Δ · ∂m/∂a
    print(f"|Δ|={escala}: real {real.item():+.3f}  lineal {lineal.item():+.3f}")
# <<<
