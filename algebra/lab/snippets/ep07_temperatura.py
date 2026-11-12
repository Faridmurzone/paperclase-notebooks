import torch
torch.manual_seed(0)
def r(t, d=2): return [round(x, d) for x in t.tolist()]
# >>>
tokens = ["el", "un", "la", "este", "mi"]
logits = torch.tensor([3.0, 2.0, 1.5, 0.5, -1.0])

for T in [0.5, 1.0, 2.0]:
    p = torch.softmax(logits / T, dim=0)       # dividir por T
    muestras = torch.multinomial(p, 1000, replacement=True)
    cuenta = torch.bincount(muestras, minlength=5)
    print(f"T={T}: p      = {r(p)}")
    print(f"       sorteos = {cuenta.tolist()}")
# <<<
