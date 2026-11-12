import json
import torch
from transformers import AutoConfig
from huggingface_hub import hf_hub_download
from safetensors import safe_open
REPO = "google/gemma-2-2b"
mapa = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))["weight_map"]
# >>>
cfg = AutoConfig.from_pretrained(REPO)
print("heads de query:", cfg.num_attention_heads,
      "· heads de key/value:", cfg.num_key_value_heads,
      "· head_dim:", cfg.head_dim)
for p in ["q_proj", "k_proj", "v_proj", "o_proj"]:
    n = f"model.layers.0.self_attn.{p}.weight"
    with safe_open(hf_hub_download(REPO, mapa[n]), framework="pt") as f:
        print(f"{p}: {f.get_slice(n).get_shape()}")
K = torch.randn(5, 4, 256)                    # T=5, 4 heads de key
K8 = K.repeat_interleave(2, dim=1)            # cada una, para 2 queries
print("K", tuple(K.shape), "→", tuple(K8.shape))
# <<<
