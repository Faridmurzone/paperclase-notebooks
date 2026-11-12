import torch
from collections import Counter
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", dtype=torch.float32)
# >>>
grupos = Counter()
for nombre, p in model.named_parameters():
    if "embed_tokens" in nombre:  grupos["embeddings (W_E = W_U)"] += p.numel()
    elif "self_attn" in nombre:   grupos["atención (Q, K, V, O)"] += p.numel()
    elif "mlp" in nombre:         grupos["MLP (gate, up, down)"] += p.numel()
    else:                         grupos["normalizaciones"] += p.numel()

total = sum(grupos.values())
for g, n in grupos.most_common():
    print(f"{g:24} {n / 1e6:8.1f} M  {n / total:6.1%}")
print(f"{'total':24} {total / 1e6:8.1f} M")
# <<<
