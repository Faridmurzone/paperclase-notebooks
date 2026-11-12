import math
from collections import Counter
from glob import glob
from huggingface_hub import snapshot_download
from safetensors import safe_open
# >>>
def grupo(nombre):
    if "router" in nombre: return "router"
    if "block_sparse_moe" in nombre: return "expertos"
    if "self_attn" in nombre: return "atención"
    return "embeddings" if "embed" in nombre else "normas"

carpeta = snapshot_download("ibm-granite/granite-3.1-1b-a400m-base",
                            local_files_only=True)
cuenta = Counter()
for archivo in glob(f"{carpeta}/*.safetensors"):
    with safe_open(archivo, "pt") as f:        # lee solo los headers
        for nombre in f.keys():
            forma = f.get_slice(nombre).get_shape()
            cuenta[grupo(nombre)] += math.prod(forma)
total = sum(cuenta.values())
activos = total - cuenta["expertos"] * (1 - 8 / 32)   # top-8 de 32
for g, n in cuenta.most_common():
    print(f"{g:<11}{n / 1e6:>8.1f} M")
print(f"total {total / 1e6:.1f} M | activos {activos / 1e6:.1f} M")
# <<<
