import copy, importlib.util, itertools, os, torch
import torch.nn.functional as F
_r = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "comun", "igpt_juguete.py")
_s = importlib.util.spec_from_file_location("igpt_juguete", _r); j = importlib.util.module_from_spec(_s); _s.loader.exec_module(j)
sft = j.cargar("ep09_sft.pt").eval()                       # el SFT del episodio 9
K = 4
PARES = list(itertools.combinations(range(K), 2))
gumbel = torch.distributions.Gumbel(0.0, 1.0)
def comparar(n):                                           # las mismas «personas» del episodio 9
    x = j.prompts(n).repeat_interleave(K, 0)
    y = j.muestrear(sft, x).view(n, K, -1)
    gusto = j.oro(y.flatten(0, 1)).view(n, K) + gumbel.sample((n, K))
    return y.gather(1, gusto.argsort(-1, descending=True)[..., None].expand_as(y))
def logp(model, y):                                        # log π(y | x), sumado sobre la respuesta
    return j.logprobs(model, y).sum(-1)
# >>>
def dpo(gamma, beta=0.5, pasos=300):
    pol = copy.deepcopy(sft).train()
    opt = torch.optim.AdamW(pol.parameters(), lr=1e-4)
    for paso in range(pasos):
        y = comparar(64)                        # rankings como los del RM
        w = torch.cat([y[:, a] for a, b in PARES])   # elegida de cada par
        l = torch.cat([y[:, b] for a, b in PARES])   # rechazada
        with torch.no_grad():
            ref_w, ref_l = logp(sft, w), logp(sft, l)   # referencia
        margen = beta * ((logp(pol, w) - ref_w) - (logp(pol, l) - ref_l))
        loss = -F.logsigmoid(margen).mean()
        loss = loss + gamma * j.nll_pre(pol, j.batch_pre(64))   # ptx
        loss.backward(); opt.step(); opt.zero_grad()
    return pol.eval()
# <<<
for gamma in [0.0, 0.1]:
    torch.manual_seed(2)
    pol = dpo(gamma)
    y = j.muestrear(pol, j.prompts(512))
    print(f"γ={gamma}  oro {j.oro(y).mean():.2f}  capacidad {j.capacidad(pol):.3f}")
    j.guardar(pol, f"ep10_dpo_{gamma}.pt")
