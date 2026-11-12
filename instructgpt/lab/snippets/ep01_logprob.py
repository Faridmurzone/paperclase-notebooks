import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
nombre = "google/gemma-2-2b"
tok = AutoTokenizer.from_pretrained(nombre)
model = AutoModelForCausalLM.from_pretrained(nombre, dtype=torch.float32)
# >>>
def logp(x, y):                                      # log p(y | x), token a token
    n = len(tok(x)["input_ids"])
    ids = tok(x + y, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        lp = model(input_ids=ids).logits[0, :-1].log_softmax(-1)
    lp_y = lp[n - 1:].gather(-1, ids[0, n:, None])   # el log-prob de cada token de y
    return lp_y.sum().item(), len(lp_y)

x = "Write a story about a wise frog."
repite = "\n\nWrite a story about a wise frog."
cuento = "\n\nOnce upon a time, there was a wise frog."
for nombre_y, y in [("repetir", repite), ("cuento", cuento)]:
    total, T = logp(x, y)
    print(f"{nombre_y:8} T={T:2d}  log p = {total:6.1f}  por token {total/T:5.2f}")
# <<<
import json, os
def por_token(x, y):
    n = len(tok(x)["input_ids"])
    ids = tok(x + y, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        lp = model(input_ids=ids).logits[0, :-1].log_softmax(-1)
    v = lp[n - 1:].gather(-1, ids[0, n:, None])[:, 0]
    return [tok.decode(i) for i in ids[0, n:]], [round(a.item(), 2) for a in v]
d = {k: dict(zip(("tokens", "logp"), por_token(x, y))) for k, y in [("repetir", repite), ("cuento", cuento)]}
json.dump(d, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep01_logprob.json"), "w"), ensure_ascii=False)
