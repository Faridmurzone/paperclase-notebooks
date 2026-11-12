import torch
torch.manual_seed(0)
d = 8
# >>>
W = torch.randn(d, d)                 # ESCRIBE en el residual stream
r_hat = torch.randn(d); r_hat /= r_hat.norm()
P = torch.eye(d) - torch.outer(r_hat, r_hat)          # I − r̂ r̂ᵀ
W_orto = P @ W                                        # W' = (I − r̂ r̂ᵀ) W

a = torch.randn(5, d)                 # entradas cualesquiera
salida = a @ W.T
ablada = salida - (salida @ r_hat)[:, None] * r_hat   # ablación en el forward
print("máx |W'a − ablación| =", f"{(a @ W_orto.T - ablada).abs().max():.1e}")
print("máx |(W'a)·r̂|       =", f"{(a @ W_orto.T @ r_hat).abs().max():.1e}")
print("rango de P:", torch.linalg.matrix_rank(P).item(), "de", d)
# <<<
