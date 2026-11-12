import copy, importlib.util, json, os, torch
_r = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "comun", "igpt_juguete.py")
_s = importlib.util.spec_from_file_location("igpt_juguete", _r); j = importlib.util.module_from_spec(_s); _s.loader.exec_module(j)
sft = j.cargar("ep07_sft.pt")
curvas = {}
# >>>
def rl(beta, gamma, iters=150):
    pol, ref, cap = copy.deepcopy(sft), copy.deepcopy(sft).eval(), []
    opt = torch.optim.AdamW(pol.parameters(), lr=3e-4)
    for it in range(iters):
        y = j.muestrear(pol, j.prompts(64))             # la política responde
        with torch.no_grad():
            kl = (j.logprobs(pol, y) - j.logprobs(ref, y)).sum(-1)
            r = j.oro(y) - beta * kl                     # recompensa − β·KL
            A = (r - r.mean()) / (r.std() + 1e-8)        # ventaja
        loss = -(A * j.logprobs(pol, y).sum(-1)).mean() # una pasada: ratio 1
        loss = loss + gamma * j.nll_pre(pol, j.batch_pre(64))  # ptx
        loss.backward(); opt.step(); opt.zero_grad()
        if it % 10 == 0 or it == iters - 1:
            cap.append(j.capacidad(pol))                 # el benchmark
    return pol, cap, j.oro(y).mean()
# <<<
for nombre, b, g in [("PPO", 0.02, 0), ("PPO-ptx", 0.02, 1.0), ("β alto", 0.5, 0)]:
    torch.manual_seed(1)
    pol, curvas[nombre], oro_final = rl(b, g)
    print(f"{nombre:8} β={b:<4} γ={g}: oro {oro_final:.2f}  capacidad {curvas[nombre][-1]:.3f}")
    j.guardar(pol, f"ep07_{nombre}.pt")
json.dump(curvas, open(os.path.join(j.DATOS, "ep07_curvas.json"), "w"))
