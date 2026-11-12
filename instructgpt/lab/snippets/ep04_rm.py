import json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
from transformers.utils import logging; logging.set_verbosity_error()
torch.manual_seed(0)
V, P, N, RUIDO = 16, 2, 5, 1.0                       # vocabulario, tokens de prompt y de respuesta
w_oculto = torch.randn(V)                             # el "gusto" de la gente, por token (sintético)
def r_real(y):
    return w_oculto[y].sum(-1)
def prompt_con_ranking():                             # K entre 4 y 9, ranking con ruido de etiquetador
    K = int(torch.randint(4, 10, (1,)))
    x = torch.randint(2, V, (P,))
    y = torch.randint(2, V, (K, N))
    nota = r_real(y) + RUIDO * torch.randn(K)
    puesto = (-nota).argsort().argsort()              # 0 = la mejor
    return torch.cat([x.expand(K, P), y], 1), puesto, r_real(y)
train = [prompt_con_ranking() for _ in range(1500)]
valid = [prompt_con_ranking() for _ in range(300)]
class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cuerpo = GPT2Model(GPT2Config(vocab_size=V, n_positions=16, n_embd=64, n_layer=2, n_head=4))
        self.cabeza = nn.Linear(64, 1)
    def forward(self, ids):
        return self.cabeza(self.cuerpo(input_ids=ids).last_hidden_state[:, -1]).squeeze(-1)
def perdida_prompt(r, puesto):
    gana = puesto[:, None] < puesto[None, :]
    return -F.logsigmoid((r[:, None] - r[None, :])[gana]).mean()
rm, B = RewardModel(), 16
opt = torch.optim.AdamW(rm.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=math.ceil(len(train) / B), eta_min=1e-4)
# >>>
forwards = comparaciones = 0
for b in range(0, len(train), B):                 # una sola epoch
    grupo = train[b:b + B]                        # B prompts, cada uno con K
    ids = torch.cat([s for s, _, _ in grupo])     # todas las respuestas juntas
    r = rm(ids).split([len(s) for s, _, _ in grupo])   # un forward por respuesta
    loss = torch.stack([perdida_prompt(ri, p)
                        for ri, (_, p, _) in zip(r, grupo)]).mean()
    loss.backward(); opt.step(); opt.zero_grad(); sched.step()
    forwards += len(ids)
    comparaciones += sum(len(s) * (len(s) - 1) // 2 for s, _, _ in grupo)
print(f"forwards {forwards:,} | pares {comparaciones:,}")
# <<<
@torch.no_grad()
def acierto(fuente):
    bien = tot = 0
    for s, p, real in valid:
        r = rm(s)
        ref = (-real).argsort().argsort() if fuente == "real" else p
        g = ref[:, None] < ref[None, :]
        bien += ((r[:, None] - r[None, :])[g] > 0).sum().item(); tot += g.sum().item()
    return bien / tot
techo = sum(((p[:, None] < p[None, :]) & (real[:, None] > real[None, :])).sum().item() for _, p, real in valid) / \
        sum((p[:, None] < p[None, :]).sum().item() for _, p, _ in valid)
a_et, a_real = acierto("etiquetas"), acierto("real")
print(f"RM vs etiquetas:     {a_et:.0%}")
print(f"etiquetador vs real: {techo:.0%}")
print(f"RM vs real:          {a_real:.0%}")
torch.save(rm.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep04_rm.pt"))
json.dump({"acierto_etiquetas": round(a_et, 3), "acierto_real": round(a_real, 3), "techo": round(techo, 3)},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep04_rm.json"), "w"))
