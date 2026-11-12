import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", dtype=torch.float32)
RAIZ = os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-scope-2b-pt-res/snapshots")
ruta = os.path.join(RAIZ, os.listdir(RAIZ)[0], "layer_12/width_16k/average_l0_82/params.npz")
# >>>
p = {k: torch.from_numpy(v) for k, v in np.load(ruta).items()}
for k in ("W_enc", "W_dec", "threshold"):
    print(k, tuple(p[k].shape))

cache = {}
capa = model.model.layers[12]
capa.register_forward_hook(lambda m, i, o: cache.update(x=o[0]))
ids = tok("The Eiffel Tower is in the city of Paris", return_tensors="pt")
model(**ids)
x = cache["x"][1:]                         # sin <bos>: [9, 2304]

pre = x @ p["W_enc"] + p["b_enc"]          # [9, 16384]
f = pre * (pre > p["threshold"])           # JumpReLU
x_hat = f @ p["W_dec"] + p["b_dec"]
var = 1 - ((x - x_hat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()
L0 = (f > 0).sum(-1)
print(f"L0 por token: {L0.tolist()}, media {L0.float().mean():.0f}")
print(f"varianza explicada: {var:.3f}")
for t in (2, 8):                           # ' Tower' y ' Paris'
    v, i = f[t].topk(3)
    nombre = tok.decode(ids.input_ids[0, t + 1])
    print(repr(nombre), list(zip(i.tolist(), v.round().tolist())))
# <<<
