import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.utils import logging; logging.set_verbosity_error()
torch.manual_seed(3)
V, P, N = 12, 2, 6
cfg = GPT2Config(vocab_size=V, n_positions=16, n_embd=64, n_layer=2, n_head=4, resid_pdrop=0, embd_pdrop=0,
                 attn_pdrop=0, bos_token_id=0, eos_token_id=0)
d = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep05_ppo_ptx.pt"))
pol, ref = GPT2LMHeadModel(cfg).eval(), GPT2LMHeadModel(cfg).eval()   # pol: la de γ = 2 del snippet anterior
pol.load_state_dict(d["pol"]); ref.load_state_dict(d["ref"])
w_rm, T = d["w_rm"], d["T"]
def rm(y):
    return w_rm[y].sum(-1)
def logp_tokens(m, ids, desde):
    lp = m(input_ids=ids).logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return lp[:, desde - 1:]
x0 = torch.randint(0, V, (4, 1)); x = [x0]
for t in range(7):
    x.append(torch.multinomial(T[x[-1][:, 0]], 1))
x_pre = torch.cat(x, 1)[:1]                         # un texto de preentrenamiento
with torch.no_grad():
    y = torch.randint(0, V, (4, P))
    for _ in range(N):
        y = torch.cat([y, torch.multinomial(pol(input_ids=y).logits[:, -1].softmax(-1), 1)], 1)
beta, gamma = 0.05, 2.0
# >>>
with torch.no_grad():
    r = rm(y[:, P:])                                   # r_θ(x, y), una por respuesta
    log_ratio = (logp_tokens(pol, y, P)
                 - logp_tokens(ref, y, P)).sum(-1)     # log π_RL(y|x) − log π_SFT(y|x)
    log_pre = logp_tokens(pol, x_pre, 1).sum()         # log π_RL(x), texto de preentreno
    objetivo = (r - beta * log_ratio).mean() + gamma * log_pre
print("r:         ", [f"{v:+.2f}" for v in r])
print("log ratio: ", [f"{v:+.2f}" for v in log_ratio])
print(f"log π(x_pre) = {log_pre:.2f}")
print(f"objetivo = {(r - beta * log_ratio).mean():.2f} + {gamma}·({log_pre:.2f})"
      f" = {objetivo:.2f}")
# <<<
import json
with torch.no_grad():
    tok_ratio = (logp_tokens(pol, y, P) - logp_tokens(ref, y, P))[0]
    tok_pre = logp_tokens(pol, x_pre, 1)[0]
json.dump({"r": [round(v, 2) for v in r.tolist()], "log_ratio": [round(v, 2) for v in log_ratio.tolist()],
           "tok_ratio": [round(v, 3) for v in tok_ratio.tolist()], "tok_pre": [round(v, 2) for v in tok_pre.tolist()],
           "log_pre": round(log_pre.item(), 2), "objetivo": round(objetivo.item(), 2)},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep05_terminos.json"), "w"))
