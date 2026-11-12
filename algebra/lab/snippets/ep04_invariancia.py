import json
import torch
from torch import nn
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
REPO = "google/gemma-2-2b"
mapa = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))["weight_map"]
archivo = hf_hub_download(REPO, mapa["model.embed_tokens.weight"])
tok = AutoTokenizer.from_pretrained(REPO)
with safe_open(archivo, framework="pt") as f:
    W_E = f.get_tensor("model.embed_tokens.weight")
# >>>
ln, rms = nn.LayerNorm(4), Gemma2RMSNorm(4)
x = torch.tensor([1.0, 2.0, 3.0, 6.0])
igual = lambda a, b: torch.allclose(a, b, atol=1e-4)
print("LayerNorm(x + 10) == LayerNorm(x):", igual(ln(x + 10), ln(x)))
print("LayerNorm(5 x)    == LayerNorm(x):", igual(ln(5 * x), ln(x)))
print("RMSNorm(5 x)      == RMSNorm(x)  :", igual(rms(5 * x), rms(x)))
print("RMSNorm(x + 10)   == RMSNorm(x)  :", igual(rms(x + 10), rms(x)))

(i,) = tok(" perro", add_special_tokens=False).input_ids
x0 = W_E[i] * 2304 ** 0.5                     # embedding real de Gemma
y = Gemma2RMSNorm(2304)(x0)
print("‖x0‖ =", round(x0.norm().item(), 1), " → ‖RMSNorm(x0)‖ =",
      round(y.norm().item(), 1), " (√2304 = 48)")
# <<<
