import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM
torch.manual_seed(0)
V, P, N = 16, 2, 6                                    # vocabulario, tokens de prompt, tokens de respuesta
w_oculto = torch.randn(V)
def rm(y):                                            # el "modelo de recompensa": el gusto oculto del ep. 3
    return w_oculto[y].sum(-1) * 0.7
cfg = LlamaConfig(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32)
ref = LlamaForCausalLM(cfg).eval()                    # π_ref: el modelo de partida (congelado)
for p_ in ref.parameters():
    p_.requires_grad_(False)

@torch.no_grad()
def muestrear(model, x, n=N):                         # muestreo token a token, temperatura 1
    for _ in range(n):
        p = model(input_ids=x).logits[:, -1].softmax(-1)
        x = torch.cat([x, torch.multinomial(p, 1)], 1)
    return x

def prompts(n):
    return torch.randint(2, V, (n, P))
pol = copy.deepcopy(ref)                              # una política que ya se movió un poco
with torch.no_grad():
    pol.lm_head.weight += 0.3 * torch.randn_like(pol.lm_head.weight)
# >>>
def logprobs(model, ids):              # log π(y_t | x, y_<t) por token
    lp = model(input_ids=ids).logits[:, :-1].log_softmax(-1)
    lp = lp.gather(-1, ids[:, 1:, None]).squeeze(-1)  # el que salió
    return lp[:, P - 1:]                   # solo la respuesta

y = muestrear(pol, prompts(2000))                    # respuestas de la política
with torch.no_grad():
    kl_tok = logprobs(pol, y) - logprobs(ref, y)     # log π − log π_ref
print("seq 0, por token:", kl_tok[0].numpy().round(2))
print(f"tokens negativos: {(kl_tok < 0).float().mean():.0%}")
print(f"estimador (promedio): {kl_tok.sum(-1).mean():.3f}")
# <<<
with torch.no_grad():                                 # KL exacta, sumando sobre el vocabulario
    lq = pol(input_ids=y).logits[:, P - 1:-1].log_softmax(-1)
    lr = ref(input_ids=y).logits[:, P - 1:-1].log_softmax(-1)
    exacta = (lq.exp() * (lq - lr)).sum(-1).sum(-1).mean()
print(f"KL exacta (promedio): {exacta:.3f}")
