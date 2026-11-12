import copy, importlib.util, json, os, torch
_r = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "comun", "igpt_juguete.py")
_s = importlib.util.spec_from_file_location("igpt_juguete", _r); j = importlib.util.module_from_spec(_s); _s.loader.exec_module(j)
sft = j.cargar("ep09_sft.pt")
curvas = {}
# >>>
def ppo_ptx(rm, sesgo, beta, gamma=1.0, iters=300):
    pol, ref, hist = copy.deepcopy(sft), copy.deepcopy(sft).eval(), []
    opt = torch.optim.AdamW(pol.parameters(), lr=3e-4)
    for it in range(iters):
        y = j.muestrear(pol, j.prompts(64))
        with torch.no_grad():
            puntaje = rm(y) - sesgo                     # paso 2: el RM juzga
            kl = (j.logprobs(pol, y) - j.logprobs(ref, y)).sum(-1)
            r = puntaje - beta * kl
            A = (r - r.mean()) / (r.std() + 1e-8)
        loss = -(A * j.logprobs(pol, y).sum(-1)).mean()
        loss = loss + gamma * j.nll_pre(pol, j.batch_pre(64))
        loss.backward(); opt.step(); opt.zero_grad()
        if it % 25 == 0 or it == iters - 1:            # RM y oro, lado a lado
            fila = [puntaje.mean(), j.oro(y).mean(), kl.mean()]
            hist.append([float(v) for v in fila])
    return pol, hist
# <<<
for nombre, rm_nombre, beta in [("RM fuerte, β=0,02", "fuerte", 0.02), ("RM débil, β=0", "débil", 0.0),
                                ("RM débil, β=0,1", "débil", 0.1)]:
    torch.manual_seed(3)
    rm, sesgo = j.cargar_rm(f"ep09_rm_{rm_nombre}.pt")
    pol, curvas[nombre] = ppo_ptx(rm, sesgo, beta)
    p, o, k = curvas[nombre][-1]
    print(f"{nombre:18} RM {p:+.2f}  oro {o:.2f}  KL {k:5.2f}")
    j.guardar(pol, f"ep09_pol_{rm_nombre}_{beta}.pt")
json.dump(curvas, open(os.path.join(j.DATOS, "ep09_curvas.json"), "w"), ensure_ascii=False)
