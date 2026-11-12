import torch
torch.set_printoptions(precision=3)

# >>>
def valle(p):
    return 0.1 * p[0] ** 2 + 3 * p[1] ** 2

H = torch.autograd.functional.hessian(valle, torch.zeros(2))
curv = torch.linalg.eigvalsh(H)          # curvatura en cada dirección
print("hessiana:", [[round(v, 3) for v in fila] for fila in H.tolist()])
print("curvaturas:", [round(c, 3) for c in curv.tolist()])
print(f"lr máximo = 2 / {curv.max():.1f} = {2 / curv.max():.3f}")
print(f"condition number = {curv.max() / curv.min():.1f}")
# <<<
