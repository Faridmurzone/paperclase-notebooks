import copy
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel
torch.manual_seed(0)
cfg = GPT2Config(vocab_size=50, n_positions=32, n_embd=32, n_layer=2, n_head=2)
pi = GPT2LMHeadModel(cfg)
beta = 0.1

def logp(m, p, y):                       # suma de log-probs de la respuesta
    ids = torch.cat([p, y], 1)
    lp = m(ids).logits[:, :-1].log_softmax(-1)
    lp = lp.gather(-1, ids[:, 1:, None]).squeeze(-1)
    return lp[:, p.shape[1] - 1:].sum(-1)

def lote(n):                             # preferidas: tokens 0–24; rechazadas: 25–49
    p = torch.randint(0, 50, (n, 4))
    return p, torch.randint(0, 25, (n, 5)), torch.randint(25, 50, (n, 5))
# >>>
ref = copy.deepcopy(pi).eval()           # π_ref congelada
opt = torch.optim.AdamW(pi.parameters(), lr=1e-3)
for paso in range(61):
    p, yw, yl = lote(32)
    with torch.no_grad():
        ref_w, ref_l = logp(ref, p, yw), logp(ref, p, yl)
    r_w = beta * (logp(pi, p, yw) - ref_w)
    r_l = beta * (logp(pi, p, yl) - ref_l)
    loss = -F.logsigmoid(r_w - r_l).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if paso % 20 == 0:
        print(f"paso {paso:2d}  loss {loss:.3f}  "
              f"r_w {r_w.mean():+.2f}  r_l {r_l.mean():+.2f}")
with torch.no_grad():                    # pares nuevos, nunca vistos
    p, yw, yl = lote(500)
    rw = logp(pi, p, yw) - logp(ref, p, yw)
    rl = logp(pi, p, yl) - logp(ref, p, yl)
print(f"r_w > r_l en 500 pares nuevos: {(rw > rl).float().mean():.0%}")
# <<<
