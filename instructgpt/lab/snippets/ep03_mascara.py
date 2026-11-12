import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.utils import logging; logging.set_verbosity_error()
torch.manual_seed(0)
cfg = GPT2Config(vocab_size=17, n_positions=32, n_embd=32, n_layer=1, n_head=2, bos_token_id=2, eos_token_id=2)
model = GPT2LMHeadModel(cfg).eval()
ids = torch.tensor([[12, 8, 15, 8, 10, 13, 1, 8, 13, 12, 8, 8, 10, 12, 13, 15, 2]])
labels = ids.clone(); labels[:, :7] = -100
# >>>
logits = model(input_ids=ids).logits[0, :-1]       # posición t predice t+1
objetivo = labels[0, 1:]
nll = F.cross_entropy(logits, objetivo, reduction="none",
                      ignore_index=-100)            # 0 donde no cuenta
cuentan = objetivo != -100
mano = nll[cuentan].sum() / cuentan.sum()           # promedio sobre R
hf = model(input_ids=ids, labels=labels).loss
print(f"tokens que cuentan: {int(cuentan.sum())} de {len(objetivo)}")
print(f"a mano {mano.item():.4f} | HF {hf.item():.4f}")

prompt_med, demo_med = 324, 18                      # medianas, tabla 11
print(f"sin máscara, el prompt sería {prompt_med/(prompt_med+demo_med):.0%}"
      " de los tokens")
# <<<
import json, os
json.dump({"objetivo": objetivo.tolist(), "nll": [round(v, 2) for v in nll.tolist()], "cuenta": cuentan.tolist()},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep03_mascara.json"), "w"))
