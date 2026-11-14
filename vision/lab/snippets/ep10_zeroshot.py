import os, sys, json
import torch
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from _clip import datos, EncImagen, EncTexto, PALABRAS
_, _, Xte, yte = datos()
ck = torch.load(os.path.join(AQUI, "..", "datos", "clip_ep10.pt"))
enc_i, enc_t = EncImagen(), EncTexto()
enc_i.load_state_dict(ck["img"]); enc_t.load_state_dict(ck["txt"])
torch.set_grad_enabled(False)
# >>>
def zero_shot(plantilla):
    clases = [plantilla.format(p=p) for p in PALABRAS]  # 10 frases
    T = enc_t(clases)                  # 10 × 32: el "clasificador"
    sims = enc_i(Xte) @ T.T            # 597 × 10 cosenos
    return (sims.argmax(1) == yte).float().mean().item(), sims

for plantilla in ["un {p}", "el número {p}", "una foto de un {p}"]:
    acc, sims = zero_shot(plantilla)
    print(f"{plantilla!r:22} precisión test: {acc:.1%}")
# <<<
_, sims = zero_shot("el número {p}")
fila = (sims[0] * ck["log_escala"].exp()).softmax(-1)
json.dump({"y0": int(yte[0]), "cos0": [round(v, 2) for v in sims[0].tolist()],
           "prob0": [round(v, 3) for v in fila.tolist()],
           "media": [[round(v, 2) for v in sims[yte == d].mean(0).tolist()] for d in range(10)]},
          open(os.path.join(AQUI, "..", "datos", "ep10_fig.json"), "w"))
