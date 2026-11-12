import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
def corto(s, n=24):                                  # una línea de salida legible
    return s.replace("\n", "⏎")[:n]
# >>>
nombre = "google/gemma-2-2b"                         # modelo base: sin ajuste
tok = AutoTokenizer.from_pretrained(nombre)
model = AutoModelForCausalLM.from_pretrained(nombre, dtype=torch.float32)

pedidos = ["Write a story about a wise frog.",       # el ejemplo del paper
           "Explain gravity to a 6 year old.",
           "Traducí al inglés: el gato duerme."]
for p in pedidos:
    ids = tok(p, return_tensors="pt")
    out = model.generate(**ids, max_new_tokens=24,
                         do_sample=False)            # greedy: lo más probable
    nuevo = tok.decode(out[0, ids["input_ids"].shape[1]:])
    print(corto(p, 32))
    print("→", corto(nuevo))
# <<<
