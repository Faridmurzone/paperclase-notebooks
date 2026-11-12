import torch
from transformers import GPT2Config, GPT2LMHeadModel
torch.manual_seed(0)
cfg = GPT2Config(vocab_size=50, n_positions=32, n_embd=32, n_layer=2, n_head=2)
modelo = GPT2LMHeadModel(cfg).eval()
prompt = torch.tensor([[5, 9, 12, 3]])
respuesta = torch.tensor([[7, 21, 8]])
# >>>
def logp_respuesta(modelo, prompt, respuesta):
    ids = torch.cat([prompt, respuesta], dim=1)      # [1, P + R]
    logits = modelo(ids).logits[:, :-1]              # posición t predice t+1
    logp = logits.log_softmax(-1)
    lp_tok = logp.gather(-1, ids[:, 1:, None]).squeeze(-1)
    P = prompt.shape[1]
    mascara = torch.zeros_like(lp_tok)
    mascara[:, P - 1:] = 1                           # solo la respuesta
    return (lp_tok * mascara).sum(-1), lp_tok[0, P - 1:]

with torch.no_grad():
    total, por_token = logp_respuesta(modelo, prompt, respuesta)
print("log p por token:", [round(x, 2) for x in por_token.tolist()])
print(f"log p(respuesta) = {total.item():.2f}")
print(f"p(respuesta)     = {total.exp().item():.2e}")
# <<<
