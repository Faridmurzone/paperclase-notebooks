import json, os
import torch
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")

# >>>
def entrenar(clip, pasos=300):
    torch.manual_seed(0)
    w = torch.full((20,), 3.0, requires_grad=True)       # la pérdida: ‖w‖²
    opt = torch.optim.Adam([w], lr=0.01)
    hist = []
    for t in range(pasos):
        opt.zero_grad()
        perdida = (w ** 2).sum()
        perdida.backward()
        if t == 50: w.grad *= 1000                        # un batch roto
        norma = torch.nn.utils.clip_grad_norm_([w], clip) # devuelve la norma de antes
        opt.step()
        est = opt.state[w]                                # √v̂ del primer peso
        rv = (est["exp_avg_sq"][0] / (1 - 0.999 ** est["step"])).sqrt()
        hist.append((perdida.item(), norma.item(), rv.item()))
    return hist
sin, con = entrenar(float("inf")), entrenar(1.0)
print("norma en el paso 50:", round(sin[50][1]))
print("paso   pérdida sin clip   con clip   √v̂ sin clip")
for t in (49, 51, 100, 299):
    print(f"{t:4}   {sin[t][0]:16.2f}   {con[t][0]:8.2f}   {sin[t][2]:11.1f}")
# <<<
json.dump({"sin": [round(h[0], 3) for h in sin], "con": [round(h[0], 3) for h in con]},
          open(os.path.join(DATOS, "ep09_pico.json"), "w"))
