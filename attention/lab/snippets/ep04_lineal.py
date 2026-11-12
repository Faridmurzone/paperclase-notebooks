import torch
torch.set_printoptions(precision=3, sci_mode=False)
d, n = 512, 100
pos = torch.arange(n).unsqueeze(1)
omega = 1 / 10000 ** (torch.arange(0, d, 2) / d)
PE = torch.zeros(n, d)
PE[:, 0::2], PE[:, 1::2] = torch.sin(pos * omega), torch.cos(pos * omega)
# >>>
def M(k):                       # una rotación de 2×2 por cada frecuencia
    c, s = torch.cos(k * omega), torch.sin(k * omega)
    R = torch.zeros(d, d)
    R[0::2, 0::2] = torch.diag(c); R[0::2, 1::2] = torch.diag(s)
    R[1::2, 0::2] = torch.diag(-s); R[1::2, 1::2] = torch.diag(c)
    return R

for k in [1, 5, 20]:
    pred = PE[:n - k] @ M(k).T            # M_k · PE(pos), todas las pos
    err = (pred - PE[k:]).abs().max()
    print(f"k={k:2}: máx |M_k PE(pos) − PE(pos+k)| = {err:.1e}")
print("bloque de M_1 para la frecuencia 0:\n", M(1)[:2, :2])
# <<<
