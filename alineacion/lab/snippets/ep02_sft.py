import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                  num_attention_heads=2, num_key_value_heads=2)
model = LlamaForCausalLM(cfg)                        # modelo de juguete, mismo vocabulario
# >>>
plantilla = "<start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n"
prompt = plantilla.format(p="¿Cuánto es dos más dos?")
respuesta = "Cuatro.<end_of_turn>"

ids_p = tok(prompt)["input_ids"]
ids = tok(prompt + respuesta, return_tensors="pt")["input_ids"]
labels = ids.clone()
labels[0, :len(ids_p)] = -100                        # el prompt no cuenta
print("tokens:", ids.shape[1], "| entrenan:", int((labels != -100).sum()))

loss_hf = model(input_ids=ids, labels=labels).loss
lg = model(input_ids=ids).logits[0, :-1]             # posición t predice t+1
loss_mano = F.cross_entropy(lg, labels[0, 1:], ignore_index=-100)
print(f"HF {loss_hf.item():.4f} | a mano {loss_mano.item():.4f}")
# <<<
assert ids[0, :len(ids_p)].tolist() == ids_p   # la frontera prompt/respuesta tokeniza igual
