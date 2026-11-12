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
W_U = W_E                        # Gemma: la misma matriz entra y sale
x = vec(" gato") + vec(" perro") # un vector de 2304, de prueba

logits = W_U @ x                 # (256000 × 2304) @ (2304) → 256000
print("logits:", tuple(logits.shape))

k = tok(" mascota", add_special_tokens=False).input_ids[0]
print("ℓ[mascota] =", round(logits[k].item(), 2),
      "= fila·x =", round((W_U[k] @ x).item(), 2))

for t in logits.topk(4).indices:     # los 4 logits más altos
    print(f"{tok.decode(int(t)):10} ℓ = {logits[t]:.2f}"
          f"   ‖fila‖ = {W_U[t].norm():.2f}")
print("‖fila‖ promedio:", round(W_U.norm(dim=1).mean().item(), 2))
# <<<
