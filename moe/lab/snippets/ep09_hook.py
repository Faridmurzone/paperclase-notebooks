import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _granite import cargar
tok, model = cargar()
# >>>
capa = model.model.layers[12].block_sparse_moe   # el MoE de la capa 12
print(type(capa.router).__name__, tuple(capa.router.weight.shape))

guardado = {}
def hook(modulo, entrada, salida):
    logits, pesos, idx = salida          # lo que devuelve el router
    guardado.update(logits=logits, pesos=pesos, idx=idx)
h = capa.router.register_forward_hook(hook)

ids = tok("for i in range(10):", return_tensors="pt")
model(**ids)
h.remove()
toks = tok.convert_ids_to_tokens(ids.input_ids[0])
print("logits", tuple(guardado["logits"].shape), "idx", tuple(guardado["idx"].shape))
t = 3                                            # el token "range"
print("expertos de", repr(toks[t]))
print(guardado["idx"][t].tolist())
print(" ".join(f"{p:.2f}"[1:] for p in guardado["pesos"][t]))
print("suman", f"{guardado['pesos'][t].sum():.2f}")
print("'for': peso máx", f"{guardado['pesos'][0].max():.2f}")
# <<<
