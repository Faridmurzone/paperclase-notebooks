import torch
from nnsight import LanguageModel
torch.manual_seed(0)
# >>>
model = LanguageModel("google/gemma-2-2b", dtype=torch.float32,
                      device_map="cpu", dispatch=True)
texto = "The Eiffel Tower is in the city of"
with model.trace(texto):                       # arma un grafo de intervención
    h12 = model.model.layers[12].output.save() # .save(): sobrevive al with
    logits = model.output.logits.save()     # salida final del modelo
print(h12.shape, logits.shape)

with model.trace(texto):
    model.model.layers[12].output[:, -1, :] *= 0.5   # intervenir = asignar
    logits2 = model.output.logits.save()
paris = model.tokenizer(" Paris", add_special_tokens=False).input_ids[0]
for l in (logits, logits2):
    print(f"p(' Paris') = {l[0, -1].softmax(-1)[paris]:.3f}")
with model.trace(texto):                       # trampa: lm_head no tiene el softcap
    crudos = model.lm_head.output.save()
print(f"lm_head: p(' Paris') = {crudos[0, -1].softmax(-1)[paris]:.3f}")
# <<<
