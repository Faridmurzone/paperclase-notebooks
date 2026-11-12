import json
from huggingface_hub import hf_hub_download
from safetensors import safe_open
REPO = "google/gemma-2-2b"
mapa = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))["weight_map"]
# >>>
nombres = ["model.embed_tokens.weight",
           "model.layers.0.self_attn.q_proj.weight",
           "model.layers.0.self_attn.k_proj.weight",
           "model.layers.0.self_attn.o_proj.weight",
           "model.layers.0.mlp.up_proj.weight",
           "model.layers.0.mlp.down_proj.weight"]
for n in nombres:
    archivo = hf_hub_download(REPO, mapa[n])
    with safe_open(archivo, framework="pt") as f:
        forma = f.get_slice(n).get_shape()   # sin cargar el tensor
    corto = n.split(".")[-2]              # q_proj, up_proj, ...
    print(f"{corto:13} {forma[0]:>6} × {forma[1]}")
# <<<
