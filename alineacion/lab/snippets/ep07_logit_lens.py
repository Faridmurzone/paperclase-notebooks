import json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
nombre = "google/gemma-2-2b"
tok = AutoTokenizer.from_pretrained(nombre)
model = AutoModelForCausalLM.from_pretrained(nombre, dtype=torch.float32)
ids = tok("The Eiffel Tower is in the city of", return_tensors="pt")
paris = tok(" Paris", add_special_tokens=False).input_ids[0]
cap = model.config.final_logit_softcapping            # 30.0 en Gemma 2
# >>>
hs = []                              # salida de cada capa, con hooks
for capa in model.model.layers:
    capa.register_forward_hook(lambda m, i, o: hs.append(o.detach()))
with torch.no_grad():
    final = model(**ids).logits[0, -1].softmax(-1)

def logit_lens(h):                  # el mismo final que usa el modelo
    l = model.lm_head(model.model.norm(h))
    return torch.tanh(l / cap) * cap             # softcap de Gemma 2

curva = []
for capa, h in enumerate(hs):
    p = logit_lens(h[0, -1]).softmax(-1)
    curva.append(p[paris].item())
    if capa % 5 == 0:
        print(f"capa {capa:2}: {tok.decode(p.argmax())!r:9} p(Paris)={p[paris]:.3f}")
print(f"modelo  : {tok.decode(final.argmax())!r:9} p(Paris)={final[paris]:.3f}")
# <<<
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "datos"), exist_ok=True)
json.dump(curva, open(os.path.join(os.path.dirname(__file__), "..", "datos", "ep07_lens.json"), "w"))
