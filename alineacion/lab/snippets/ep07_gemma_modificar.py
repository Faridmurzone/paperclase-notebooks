import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
nombre = "google/gemma-2-2b"
tok = AutoTokenizer.from_pretrained(nombre)
model = AutoModelForCausalLM.from_pretrained(nombre, dtype=torch.float32)
ids = tok("The Eiffel Tower is in the city of", return_tensors="pt")
def top3(logits):
    p = logits[0, -1].softmax(-1).topk(3)
    return [(tok.decode(i), round(v.item(), 2)) for v, i in zip(p.values, p.indices)]
# >>>
def apagar_ultimo(module, input, output):
    nuevo = output.clone()
    nuevo[:, -1, :] = input[0][:, -1, :]   # la capa no escribe en el último token
    return nuevo                            # tensor, no tupla (transformers 5.x)

with torch.no_grad():
    print("normal  ", top3(model(**ids).logits))
    hs = [model.model.layers[l].register_forward_hook(apagar_ultimo)
          for l in range(8, 20)]
    print("sin 8-19", top3(model(**ids).logits))
    for h in hs:
        h.remove()
    print("de nuevo", top3(model(**ids).logits))
# <<<
