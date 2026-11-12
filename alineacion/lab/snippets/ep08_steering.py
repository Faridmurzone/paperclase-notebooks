import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", dtype=torch.float32)
pos = ["I love this, it is wonderful.", "What a fantastic, beautiful day.", "This is the best meal I have ever had.",
       "I am so happy and grateful.", "The movie was brilliant and moving.", "Everything went perfectly, great job."]
neg = ["I hate this, it is terrible.", "What an awful, miserable day.", "This is the worst meal I have ever had.",
       "I am so sad and angry.", "The movie was boring and stupid.", "Everything went wrong, horrible job."]
CAPA = 12
def media(frases):
    hs = [model(**tok(f, return_tensors="pt"), output_hidden_states=True).hidden_states[CAPA + 1][0, 1:].mean(0)
          for f in frases]                       # sin el <bos>
    return torch.stack(hs).mean(0)
# >>>
r = media(pos) - media(neg)                 # «positivo − negativo»
print(f"|r| = {r.norm():.1f}")

def sumar(alfa):
    return lambda m, i, o: o + alfa * r     # x + α r, toda posición

ids = tok("I think that this restaurant is", return_tensors="pt")
for alfa in (-4, 0, 4):
    h = model.model.layers[CAPA].register_forward_hook(sumar(alfa))
    g = model.generate(**ids, max_new_tokens=9, do_sample=False)
    h.remove()
    print(f"{alfa:+d}", repr(tok.decode(g[0, ids.input_ids.shape[1]:])))
# <<<
