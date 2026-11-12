import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
nombre = "google/gemma-2-2b"
tok = AutoTokenizer.from_pretrained(nombre)
model = AutoModelForCausalLM.from_pretrained(nombre, dtype=torch.float32)
def corto(s, n=31):
    return s.strip().replace("\n", "⏎")[:n]
# >>>
torch.manual_seed(0)
x = "Write a story about a wise frog."
ids = tok(x, return_tensors="pt")
out = model.generate(**ids, max_new_tokens=20, do_sample=True,
                     temperature=1.0,                # muestrear, no elegir
                     num_return_sequences=4)
for k, fila in enumerate(out):
    nuevo = tok.decode(fila[ids["input_ids"].shape[1]:],
                       skip_special_tokens=True)
    print(k + 1, corto(nuevo))
# <<<
