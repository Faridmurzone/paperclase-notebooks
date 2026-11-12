import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
nombre = "google/gemma-2-2b"
tok = AutoTokenizer.from_pretrained(nombre)
model = AutoModelForCausalLM.from_pretrained(nombre, dtype=torch.float32)
ids = tok("The Eiffel Tower is in the city of", return_tensors="pt")
# >>>
cache = {}
def guardar(module, args, kwargs, output):
    cache["out"] = output.detach()
    cache["kwargs"] = sorted(kwargs)

capa = model.model.layers[12]                 # Gemma2DecoderLayer
h = capa.register_forward_hook(guardar, with_kwargs=True)
with torch.no_grad():
    out = model(**ids, output_hidden_states=True)
h.remove()

print(type(cache["out"]).__name__, tuple(cache["out"].shape))
print(cache["kwargs"][:4])
print(torch.equal(cache["out"], out.hidden_states[13]))
print(f"norma por posición: {cache['out'][0].norm(dim=-1).round().tolist()}")
# <<<
