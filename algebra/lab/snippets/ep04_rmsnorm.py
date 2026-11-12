import json
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
REPO = "google/gemma-2-2b"
mapa = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))["weight_map"]
# >>>
x = torch.tensor([1.0, 2.0, 3.0, 6.0])

rms = torch.sqrt((x ** 2).mean() + 1e-6)      # raíz del promedio de x²
print("RMS =", round(rms.item(), 3))
print("x / RMS      :", (x / rms).numpy().round(3))

norma = Gemma2RMSNorm(4)                      # la de Gemma 2
print("Gemma2RMSNorm:", norma(x).detach().numpy().round(3))
print("weight inicial:", norma.weight.data.numpy())   # usa 1 + w

nombre = "model.norm.weight"                   # la norma final de Gemma
with safe_open(hf_hub_download(REPO, mapa[nombre]), "pt") as f:
    w = f.get_tensor(nombre)
print("norma final:", tuple(w.shape), " 1 + w va de",
      round(1 + w.min().item(), 2), "a", round(1 + w.max().item(), 2))
# <<<
