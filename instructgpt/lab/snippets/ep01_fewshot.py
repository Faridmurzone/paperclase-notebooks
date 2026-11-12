import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
nombre = "google/gemma-2-2b"
tok = AutoTokenizer.from_pretrained(nombre)
model = AutoModelForCausalLM.from_pretrained(nombre, dtype=torch.float32)
import textwrap
# >>>
prefijo = ("Instruction: Translate to Spanish: good morning.\n"
           "Answer: Buenos días.\n\n"
           "Instruction: Name three colors.\n"
           "Answer: Red, green and blue.\n\n")
pedido = "Instruction: Write a story about a wise frog.\nAnswer:"

ids = tok(prefijo + pedido, return_tensors="pt")
out = model.generate(**ids, max_new_tokens=40, do_sample=False)
nuevo = tok.decode(out[0, ids["input_ids"].shape[1]:])
for linea in textwrap.wrap(nuevo.split("\n")[0], 34)[:5]:
    print(linea)
# <<<
