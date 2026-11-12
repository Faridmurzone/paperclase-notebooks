import json
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer
REPO = "google/gemma-2-2b"
mapa = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))["weight_map"]
tok = AutoTokenizer.from_pretrained(REPO)
with safe_open(hf_hub_download(REPO, mapa["model.embed_tokens.weight"]), "pt") as f:
    W_E = f.get_tensor("model.embed_tokens.weight")
def vec(p):
    (i,) = tok(p, add_special_tokens=False).input_ids
    return W_E[i]
# >>>
pares = [(" man", " woman"), (" he", " she"), (" his", " her"),
         (" boy", " girl"), (" father", " mother")]
r = torch.stack([vec(a) - vec(b) for a, b in pares]).mean(0)
r_hat = r / r.norm()                       # la dirección, unitaria

P = torch.eye(2304) - torch.outer(r_hat, r_hat)   # I − r̂ r̂ᵀ
for p in [" king", " queen", " actor", " actress"]:
    antes = (vec(p) @ r_hat).item()
    despues = round(((P @ vec(p)) @ r_hat).item(), 2) + 0.0
    print(f"{p:9} x·r̂ = {antes:+.2f}   después: {despues:+.2f}")

cos = lambda a, b: (a @ b / (a.norm() * b.norm())).item()
k, q = vec(" king"), vec(" queen")
print("cos(king, queen):", round(cos(k, q), 2),
      "→", round(cos(P @ k, P @ q), 2))
# <<<
