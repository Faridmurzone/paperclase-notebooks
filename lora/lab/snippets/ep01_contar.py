import glob, json, os
from safetensors import safe_open
d = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/*/"))[0]
indice = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
def forma(nombre):                          # lee solo el encabezado, no los números
    with safe_open(d + indice[nombre], "pt") as f:
        return f.get_slice(nombre).get_shape()
# >>>
grupos = {"embedding": 0, "atención": 0, "MLP": 0, "normas": 0}
for nombre in indice:                       # 288 tensores con nombre
    n = 1
    for lado in forma(nombre):              # p. ej. [2048, 2304]
        n *= lado
    if "embed" in nombre:      grupos["embedding"] += n
    elif "self_attn" in nombre: grupos["atención"] += n
    elif "mlp" in nombre:      grupos["MLP"] += n
    else:                      grupos["normas"] += n

total = sum(grupos.values())
for g, n in grupos.items():
    print(f"{g:10} {n:>13,}  {n / total:6.1%}")
print(f"{'total':10} {total:>13,}")
# <<<
json.dump(dict(grupos, total=total), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep01_conteo.json"), "w"))
