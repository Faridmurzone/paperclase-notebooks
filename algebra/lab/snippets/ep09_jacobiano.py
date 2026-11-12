import torch
from torch.autograd.functional import jacobian
def r(t): return [[round(v, 3) for v in f] for f in t.tolist()] if t.dim() == 2 else [round(v, 3) for v in t.tolist()]
# >>>
W = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
f = lambda x: W @ x                       # R² → R²: lineal
g = lambda u: torch.tanh(u)               # R² → R²: por elemento
x = torch.tensor([0.3, -0.2])

Jf = jacobian(f, x)                       # es W
Jg = jacobian(g, f(x))                    # diagonal: 1 - tanh²
Jgf = jacobian(lambda x: g(f(x)), x)
print("J_f =", Jf.tolist())
print("J_g =", r(Jg))
print("J_g @ J_f == J_(g∘f):", torch.allclose(Jg @ Jf, Jgf))

v = torch.tensor([1.0, 0.0])              # backward = v^T J
xr = x.clone().requires_grad_()
g(f(xr)).backward(v)
print("v^T J      =", r(xr.grad))
print("fila 0 de J =", r(Jgf[0]))
# <<<
