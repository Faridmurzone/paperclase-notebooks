import copy, json, os
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel
from transformers.utils import logging; logging.set_verbosity_error()
torch.manual_seed(0)
V, P, N = 12, 2, 6                                   # vocabulario, tokens de prompt, de respuesta
T = torch.full((V, V), 0.02)                         # "lenguaje" de preentrenamiento: cadena de Markov
for a in range(V):
    for b in torch.randperm(V)[:2]:
        T[a, b] = 1.0
T = T / T.sum(1, keepdim=True)
def texto(n, L=8):
    x = torch.zeros(n, L, dtype=torch.long); x[:, 0] = torch.randint(0, V, (n,))
    for t in range(1, L):
        x[:, t] = torch.multinomial(T[x[:, t - 1]], 1)[:, 0]
    return x
w_rm = torch.randn(V)                                # el RM: un gusto por token que no respeta el lenguaje
def rm(y):
    return w_rm[y].sum(-1)
cfg = GPT2Config(vocab_size=V, n_positions=16, n_embd=64, n_layer=2, n_head=4, resid_pdrop=0, embd_pdrop=0,
                 attn_pdrop=0, bos_token_id=0, eos_token_id=0)
base = GPT2LMHeadModel(cfg)                          # preentrenar = aprender la cadena de Markov
o = torch.optim.AdamW(base.parameters(), lr=3e-3)
for _ in range(400):
    x = texto(64); base(input_ids=x, labels=x).loss.backward(); o.step(); o.zero_grad()
ref = copy.deepcopy(base).eval()                     # π_SFT: acá, el modelo de partida (congelado)
for p_ in ref.parameters():
    p_.requires_grad_(False)
val_pre = texto(500)
@torch.no_grad()
def muestrear(m, x):
    for _ in range(N):
        x = torch.cat([x, torch.multinomial(m(input_ids=x).logits[:, -1].softmax(-1), 1)], 1)
    return x
def logprobs(m, ids):
    lp = m(input_ids=ids).logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return lp[:, P - 1:]
def perdida_ppo(lp, lp_old, A, eps=0.2):
    ratio = (lp - lp_old).exp()
    return -torch.min(ratio * A[:, None], ratio.clamp(1 - eps, 1 + eps) * A[:, None]).mean()
@torch.no_grad()
def ce_preentreno(m):
    return m(input_ids=val_pre, labels=val_pre).loss.item()
historia = {}
def nueva_politica():                                # misma semilla y mismo punto de partida para cada γ
    torch.manual_seed(1)
    pol = copy.deepcopy(base).train()
    return pol, torch.optim.AdamW(pol.parameters(), lr=1e-3)
def anotar(gamma, it, y, pol):
    historia.setdefault(gamma, []).append((it, rm(y[:, P:]).mean().item(), ce_preentreno(pol)))
# >>>
def entrenar(gamma, beta=0.05, iters=60):
    pol, opt = nueva_politica()                       # parte del modelo base
    for it in range(iters):
        y = muestrear(pol, texto(64, P))              # 1. la política responde
        with torch.no_grad():
            lp_old, lp_ref = logprobs(pol, y), logprobs(ref, y)
            R = rm(y[:, P:]) - beta * (lp_old - lp_ref).sum(-1)  # 2. r − β·KL
            A = (R - R.mean()) / (R.std() + 1e-8)     # 3. ventaja, sin crítico
        for mb in torch.arange(64).split(32):         # 4. minibatches, 1 pasada
            perdida_ppo(logprobs(pol, y[mb]), lp_old[mb], A[mb]).backward()
            z = texto(64)                             # texto de preentrenamiento
            (gamma * pol(input_ids=z, labels=z).loss).backward()  # + γ·∇ptx
            opt.step(); opt.zero_grad()
        anotar(gamma, it, y, pol)
    return pol

print(f"antes de RL:  CE {ce_preentreno(base):.2f}")
for gamma in [0, 0.5, 2]:
    pol = entrenar(gamma)
    _, r, ce = historia[gamma][-1]
    print(f"γ = {gamma:<3}  RM {r:+.1f}  CE {ce:.2f}")
# <<<
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
json.dump({str(g): h for g, h in historia.items()} | {"ce_base": ce_preentreno(base), "rm_max": N * w_rm.max().item()},
          open(os.path.join(D, "ep05_ppo_ptx.json"), "w"))
torch.save({"pol": pol.state_dict(), "ref": ref.state_dict(), "w_rm": w_rm, "T": T}, os.path.join(D, "ep05_ppo_ptx.pt"))
