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
# >>>
(i,) = tok(" perro", add_special_tokens=False).input_ids
print("id de ' perro':", i)

uno = torch.zeros(256_000)
uno[i] = 1.0                         # one-hot: un 1 y 255.999 ceros
por_matriz = uno @ W_E               # (256000) @ (256000 × 2304)
buscar = W_E[i]                      # la fila i, sin cuentas
print("forma:", tuple(por_matriz.shape))
print("iguales:", torch.equal(por_matriz, buscar))

x0 = buscar * 2304 ** 0.5            # Gemma la escala por √d = 48
print("‖fila‖ =", round(buscar.norm().item(), 2),
      " ‖x0‖ =", round(x0.norm().item(), 1))
# <<<
