import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
nombre = "google/gemma-2-2b"
tok = AutoTokenizer.from_pretrained(nombre)
model = AutoModelForCausalLM.from_pretrained(nombre, dtype=torch.float32)
# >>>
x = "Write a story about a wise frog."
ids = tok(x, return_tensors="pt")
with torch.no_grad():
    logits = model(**ids).logits[0, -1]              # [256000]: un logit por token
p = logits.softmax(-1)                               # probabilidades que suman 1
top = p.topk(5)
for prob, i in zip(top.values, top.indices):
    print(f"{tok.decode(i)!r:12} {prob:.3f}")
# <<<
import json, os
json.dump({"labels": [tok.decode(i) for i in top.indices], "logits": [round(logits[i].item(), 2) for i in top.indices],
           "probs": [round(v.item(), 3) for v in top.values]},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep01_proximo.json"), "w"), ensure_ascii=False)
