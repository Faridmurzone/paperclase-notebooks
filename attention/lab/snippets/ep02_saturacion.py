import torch
torch.manual_seed(0)
# >>>
def medir(d_k, escalar):
    q = torch.randn(1000, 1, d_k)            # 1000 consultas
    k = torch.randn(1000, 8, d_k)            # 8 claves cada una
    s = (q * k).sum(-1)                      # [1000, 8] productos punto
    if escalar:
        s = s / d_k ** 0.5
    s.requires_grad_(True)
    a = s.softmax(-1)
    a[:, 0].sum().backward()                 # ∂ peso del token 0 / ∂ scores
    return a.max(-1).values.mean(), s.grad.abs().mean()

for d_k in [4, 64, 512]:
    for esc in (False, True):
        pmax, g = medir(d_k, esc)
        print(f"d_k={d_k:3} {'÷√d_k' if esc else 'crudo'}:"
              f" peso máx {pmax:.2f}  |grad| {g:.4f}")
# <<<
