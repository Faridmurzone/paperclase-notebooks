import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
# >>>
nombre = "google/gemma-2-2b"
tok = AutoTokenizer.from_pretrained(nombre)
model = AutoModelForCausalLM.from_pretrained(nombre, dtype=torch.float32)

ids = tok("La capital de Francia es", return_tensors="pt")
with torch.no_grad():
    out = model(**ids, output_hidden_states=True)

print(len(out.hidden_states), out.hidden_states[0].shape)  # embeddings + 26 capas
print(out.logits.shape)                                     # [1, pos, vocabulario]
p = out.logits[0, -1].softmax(-1)
top = p.topk(5)
for prob, i in zip(top.values, top.indices):
    print(f"{tok.decode(i)!r:12} {prob:.3f}")
paris = tok(" París", add_special_tokens=False)["input_ids"][0]
print("' París'", f"{p[paris]:.3f}", "puesto", int((p > p[paris]).sum()) + 1)
# <<<
