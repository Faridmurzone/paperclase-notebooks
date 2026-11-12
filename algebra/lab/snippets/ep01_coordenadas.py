import json
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer
REPO = "google/gemma-2-2b"
mapa = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))["weight_map"]
archivo = hf_hub_download(REPO, mapa["model.embed_tokens.weight"])
tok = AutoTokenizer.from_pretrained(REPO)
with safe_open(archivo, framework="pt") as f:
    W_E = f.get_tensor("model.embed_tokens.weight")
def vec(p):
    (i,) = tok(p, add_special_tokens=False).input_ids
    return W_E[i]
# >>>
perro, gato = vec(" perro"), vec(" gato")

productos = perro * gato       # 2304 productos, sin sumar
print("cuántos   :", productos.numel())
top = productos.abs().topk(6).indices  # los 6 que más pesan
print("coord.    :", top.tolist())
print("productos :", productos[top].numpy().round(3))
resto = productos.sum() - productos[top].sum()
print("el resto  :", round(resto.item(), 3))
print("suma      :", round(productos.sum().item(), 3))
print("perro@gato:", round((perro @ gato).item(), 3))
print("‖perro‖   :", round(perro.norm().item(), 3))
print("‖gato‖    :", round(gato.norm().item(), 3))
cos = perro @ gato / (perro.norm() * gato.norm())
print("coseno    :", round(cos.item(), 3))
# <<<
