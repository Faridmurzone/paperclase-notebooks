import json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
nombre = "google/gemma-2-2b"
tok = AutoTokenizer.from_pretrained(nombre)
model = AutoModelForCausalLM.from_pretrained(nombre, dtype=torch.float32)
capas = model.model.layers
PARIS, ROMA = (tok(w, add_special_tokens=False).input_ids[0] for w in (" Paris", " Rome"))
torch.set_grad_enabled(False)
# >>>
limpio = tok("The Eiffel Tower is in the city of", return_tensors="pt")
corrupto = tok("The Colosseum is in the city of", return_tensors="pt")
print(limpio.input_ids.shape, corrupto.input_ids.shape)   # mismo largo
ld = lambda logits: (logits[0, -1, PARIS] - logits[0, -1, ROMA]).item()

h_limpio = []                                   # 1) correr limpio y guardar
hs = [c.register_forward_hook(lambda m, i, o: h_limpio.append(o.clone()))
      for c in capas]
ld_limpio = ld(model(**limpio).logits); [h.remove() for h in hs]
ld_corr = ld(model(**corrupto).logits)          # 2) correr corrupto

def parchear(capa, pos):                        # 3) corrupto + un pedazo limpio
    def pegar(m, i, o):
        o[:, pos] = h_limpio[capa][:, pos]; return o
    h = capas[capa].register_forward_hook(pegar)
    efecto = ld(model(**corrupto).logits); h.remove()
    return (efecto - ld_corr) / (ld_limpio - ld_corr)

tabla = {pos: [parchear(c, pos) for c in range(26)] for pos in (3, 8)}
print(f"logit diff limpio {ld_limpio:.2f}, corrupto {ld_corr:.2f}")
for c in (0, 5, 10, 15, 20, 25):
    print(f"capa {c:2}  pos 'Tower' {tabla[3][c]:5.2f}   pos 'of' {tabla[8][c]:5.2f}")
# <<<
json.dump(tabla, open(os.path.join(os.path.dirname(__file__), "..", "datos", "ep07_patching.json"), "w"))
