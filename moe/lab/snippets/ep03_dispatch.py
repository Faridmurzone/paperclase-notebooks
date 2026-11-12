import torch
torch.manual_seed(3)
# >>>
T, k, E = 6, 2, 4
idx = torch.stack([torch.randperm(E)[:k] for _ in range(T)])  # [T, k]
print("idx por token:", idx.tolist())
tok = torch.arange(T).repeat_interleave(k)    # de qué token es cada fila
exp = idx.reshape(-1)                         # a qué experto va
orden = exp.argsort(stable=True)              # agrupar por experto
tok, exp = tok[orden], exp[orden]
cuenta = torch.bincount(exp, minlength=E)     # cuántos por experto
for e, filas in enumerate(tok.split(cuenta.tolist())):
    print(f"experto {e} ({cuenta[e]}): tokens {filas.tolist()}")
# <<<
