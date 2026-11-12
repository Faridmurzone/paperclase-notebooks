import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", dtype=torch.float32)
model.requires_grad_(False)                       # sin gradientes de pesos: ahorra ~10 GB
capas = model.model.layers
PARIS, ROMA = (tok(w, add_special_tokens=False).input_ids[0] for w in (" Paris", " Rome"))
limpio = tok("The Eiffel Tower is in the city of", return_tensors="pt")
corrupto = tok("The Colosseum is in the city of", return_tensors="pt")
ld = lambda logits: logits[0, -1, PARIS] - logits[0, -1, ROMA]
def correr(ids, guardar=None, pegar=None):
    hs = []
    if guardar is not None:
        hs.append(model.model.embed_tokens.register_forward_hook(lambda m, i, o: o.requires_grad_()))
        for c in capas:
            hs.append(c.register_forward_hook(lambda m, i, o: (o.retain_grad(), guardar.append(o))[0]))
    if pegar is not None:
        capa, pos, valor = pegar
        def h(m, i, o):
            o = o.clone(); o[:, pos] = valor; return o
        hs.append(capas[capa].register_forward_hook(h))
    out = ld(model(**ids).logits)
    for h in hs: h.remove()
    return out
# >>>
a_limpio, a_corr = [], []
ld_limpio = correr(limpio, guardar=a_limpio).item()
m = correr(corrupto, guardar=a_corr)               # una pasada hacia adelante
m.backward()                                       # y UNA hacia atrás
ld_corr, escala = m.item(), ld_limpio - m.item()

def atribucion(capa, pos):                         # (a_limpio − a_corr) · ∂m/∂a
    delta = a_limpio[capa][0, pos] - a_corr[capa][0, pos]
    return (delta @ a_corr[capa].grad[0, pos]).item() / escala

with torch.no_grad():
    for capa in (5, 15, 20):
        for pos, nombre in ((3, "Tower"), (8, "of")):
            real = (correr(corrupto, pegar=(capa, pos, a_limpio[capa][0, pos]))
                    - ld_corr) / escala
            est = atribucion(capa, pos)
            print(f"capa {capa:2} {nombre:5}  estimado {est:5.2f}   real {real:5.2f}")
# <<<
