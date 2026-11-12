import json
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer
REPO = "google/gemma-2-2b"
mapa = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))["weight_map"]
archivo = hf_hub_download(REPO, mapa["model.embed_tokens.weight"])
# >>>
tok = AutoTokenizer.from_pretrained(REPO)
with safe_open(archivo, framework="pt") as f:          # solo esa matriz
    W_E = f.get_tensor("model.embed_tokens.weight")    # 256000 × 2304
print("W_E:", tuple(W_E.shape))

def vec(palabra):                                      # su fila de W_E
    (i,) = tok(palabra, add_special_tokens=False).input_ids
    return W_E[i]

def coseno(u, v):
    return (u @ v / (u.norm() * v.norm())).item()

for a, b in [(" perro", " gato"), (" perro", " auto"),
             (" rey", " reina"), (" lunes", " martes")]:
    print(f"{a:7} ·{b:7}  cos = {coseno(vec(a), vec(b)):+.2f}")
# <<<
