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
def acts(frases):
    return [model(**tok(f, return_tensors="pt"), output_hidden_states=True).hidden_states[CAPA + 1][0, 1:].mean(0)
            for f in frases]
A_pos, A_neg = torch.stack(acts(pos)), torch.stack(acts(neg))
def continuar(ids):
    g = model.generate(**ids, max_new_tokens=8, do_sample=False)
    return repr(tok.decode(g[0, ids.input_ids.shape[1]:]))
# >>>
r = A_pos.mean(0) - A_neg.mean(0)
r_hat = r / r.norm()                                  # vector unitario
print(f"x·r̂ medio: positivas {(A_pos @ r_hat).mean():.1f}, "
      f"negativas {(A_neg @ r_hat).mean():.1f}")

def ablar(m, i, x):
    return x - (x @ r_hat)[..., None] * r_hat         # x − (x·r̂) r̂

ids = tok("The food was awful and the waiter was", return_tensors="pt")
print("normal :", continuar(ids))
hs = [c.register_forward_hook(ablar) for c in model.model.layers]
print("ablada :", continuar(ids))
leer = model.model.layers[CAPA].register_forward_hook(   # corre después
    lambda m, i, x: print(f"x·r̂ en capa {CAPA}: {(x[0, -1] @ r_hat):.1e}"))
model(**ids)
for h in hs + [leer]: h.remove()
# <<<
