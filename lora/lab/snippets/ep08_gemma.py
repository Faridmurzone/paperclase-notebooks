import glob
import json
import os
import torch
from safetensors import safe_open
d = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/*/"))[0]
indice = json.load(open(d + "model.safetensors.index.json"))["weight_map"]


def cargar(nombre):
    with safe_open(d + indice[nombre], "pt") as f:
        return f.get_tensor(nombre)


N = torch.distributions.Normal(0, 1)
_neg = N.icdf(torch.linspace(0.03, 0.5, 8))
_pos = N.icdf(torch.linspace(0.5, 0.97, 9))
NF4 = torch.cat([_neg, _pos[1:]])
NF4 = NF4 / NF4.abs().max()
INT4 = torch.arange(-7, 8) / 7


def cuantizar(W, niveles, bloque):
    w = W.reshape(-1, bloque)                         # un bloque por fila
    absmax = w.abs().max(dim=1, keepdim=True).values
    codigo = ((w / absmax)[..., None] - niveles).abs().argmin(-1)
    return (niveles[codigo] * absmax).reshape(W.shape)


# >>>
W = cargar("model.layers.12.self_attn.q_proj.weight").float()
print("W:", tuple(W.shape))
print(f"desvío {W.std():.4f}, máx {W.abs().max():.3f}")

def error(W_hat):                            # error relativo de Frobenius
    return ((W - W_hat).norm() / W.norm()).item()

print(f"int4, un absmax para todo: {error(cuantizar(W, INT4, W.numel())):.1%}")
print(f"int4, bloques de 64:       {error(cuantizar(W, INT4, 64)):.1%}")
print(f"NF4,  bloques de 64:       {error(cuantizar(W, NF4, 64)):.1%}")
# <<<
