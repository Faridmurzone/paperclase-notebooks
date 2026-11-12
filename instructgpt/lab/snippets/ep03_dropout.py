import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.utils import logging; logging.set_verbosity_error()
torch.manual_seed(0)
# >>>
cfg = GPT2Config(vocab_size=17, n_positions=32, n_embd=64, n_layer=2,
                 n_head=4, resid_pdrop=0.2,          # residual dropout
                 embd_pdrop=0.0, attn_pdrop=0.0)
model = GPT2LMHeadModel(cfg)
print(model.transformer.h[0].attn.resid_dropout)     # antes de sumar
print(model.transformer.h[0].mlp.dropout)            # al residual stream

ids = torch.tensor([[12, 8, 15, 8, 10, 13, 1]])
model.train()
a, b = model(input_ids=ids).logits, model(input_ids=ids).logits
print("train: dos forwards iguales?", torch.allclose(a, b))
model.eval()
a, b = model(input_ids=ids).logits, model(input_ids=ids).logits
print("eval:  dos forwards iguales?", torch.allclose(a, b))
# <<<
