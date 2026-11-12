import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import *
torch.manual_seed(0)
import json
c = torch.load(os.path.join(DATOS, "ep02_completo.pt"))
W0, dW = c["W0"], c["dW"]
# >>>
def rango_k(d, k):                                     # lo mejor de rango k
    U, S, Vt = torch.linalg.svd(d, full_matrices=False)
    return U[:, :k] @ torch.diag(S[:k]) @ Vt[:k]

res = {}
for k in [0, 1, 2, 4, 8, 16, 32]:
    red = cargar_base()
    red.load_state_dict({n: p for n, p in c["red"].items() if "bias" in n},
                        strict=False)                  # sesgos del ajuste
    for j, i in enumerate((0, 2, 4)):
        red[i].weight.data = W0[j] + rango_k(dW[j], k)
    res[k] = precision(red, espejo(Xte), yte)
    print(f"ΔW de rango {k:2}: {res[k]:.1%}")
# <<<
json.dump(res, open(os.path.join(DATOS, "ep02_truncar.json"), "w"))
