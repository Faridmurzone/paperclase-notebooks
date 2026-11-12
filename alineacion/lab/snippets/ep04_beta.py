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

def logprobs(model, ids):                             # log π(y_t | x, y_<t) de cada token de la respuesta
    lp = model(input_ids=ids).logits[:, :-1].log_softmax(-1)
    lp = lp.gather(-1, ids[:, 1:, None]).squeeze(-1)  # el log-prob del token que salió
    return lp[:, P - 1:]                              # solo las N posiciones de la respuesta

def prompts(n):
    return torch.randint(2, V, (n, P))

def recompensa(score, kl_tok, beta=0.1):
    r = -beta * kl_tok
    r[:, -1] += score
    return r

def gae(r, v, gamma=1.0, lam=0.95):
    A, ultimo, T = torch.zeros_like(r), 0.0, r.shape[1]
    for t in reversed(range(T)):
        v_sig = v[:, t + 1] if t + 1 < T else 0.0
        delta = r[:, t] + gamma * v_sig - v[:, t]
        ultimo = delta + gamma * lam * ultimo
        A[:, t] = ultimo
    return A, A + v

def perdida_ppo(lp, lp_old, A, eps=0.2):
    ratio = (lp - lp_old).exp()
    return -torch.min(ratio * A, ratio.clamp(1 - eps, 1 + eps) * A).mean()

class Critico(nn.Module):                             # V(s_t): una cabeza escalar por posición
    def __init__(self):
        super().__init__()
        self.cuerpo = LlamaForCausalLM(cfg).model
        self.cabeza = nn.Linear(cfg.hidden_size, 1)
    def forward(self, ids):
        h = self.cuerpo(input_ids=ids).last_hidden_state[:, P - 1:-1]
        return self.cabeza(h).squeeze(-1)

def rlhf(beta, iters=61):
    torch.manual_seed(0)
    pol, critico = copy.deepcopy(ref).train(), Critico()
    for p_ in pol.parameters():
        p_.requires_grad_(True)
    opt = torch.optim.AdamW([*pol.parameters(), *critico.parameters()], lr=1e-3)
    for it in range(iters):
        y = muestrear(pol, prompts(64))
        with torch.no_grad():
            lp_old, lp_ref = logprobs(pol, y), logprobs(ref, y)
            score = rm(y[:, P:])
            A, R = gae(recompensa(score, lp_old - lp_ref, beta), critico(y))
            A = (A - A.mean()) / (A.std() + 1e-8)
        for _ in range(4):
            loss = perdida_ppo(logprobs(pol, y), lp_old, A) + 0.5 * ((critico(y) - R) ** 2).mean()
            loss.backward(); opt.step(); opt.zero_grad()
    y = muestrear(pol, prompts(1000))
    with torch.no_grad():
        kl = (logprobs(pol, y) - logprobs(ref, y)).sum(-1).mean()
    return rm(y[:, P:]).mean().item(), kl.item(), y[0, P:].tolist()
# >>>
for beta in [0.0, 0.1, 0.3, 1.0]:
    score, kl, ejemplo = rlhf(beta)
    print(f"β={beta:<4} RM {score:+.2f}  KL {kl:5.2f}  {ejemplo}")
# <<<
print("token preferido por el RM:", int(w_oculto.argmax()))
