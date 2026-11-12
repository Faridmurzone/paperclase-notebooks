import json, os
import torch
def grad(p): return torch.stack([0.2 * p[0], 6 * p[1]])
def perdida(p): return 0.1 * p[0]**2 + 3 * p[1]**2
# >>>
p = torch.tensor([-4.5, 1.5])
m, v = torch.zeros(2), torch.zeros(2)
b1, b2, eta, eps = 0.9, 0.999, 0.1, 1e-8
for t in range(1, 61):
    g = grad(p)
    m = b1 * m + (1 - b1) * g             # promedio móvil de g
    v = b2 * v + (1 - b2) * g**2          # promedio móvil de g²
    paso = eta * (m / (1 - b1**t)) / ((v / (1 - b2**t)).sqrt() + eps)
    p = p - paso                          # Adam, en una línea
    if t == 1:
        print("gradiente 1:", [round(x, 2) for x in g.tolist()],
              " paso 1:", [round(x, 3) for x in paso.tolist()])
print(f"a mano, 60 pasos: pérdida {perdida(p):.4f}")

q = torch.tensor([-4.5, 1.5], requires_grad=True)
opt = torch.optim.Adam([q], lr=0.1)
for _ in range(60):
    opt.zero_grad(); perdida(q).backward(); opt.step()
print("igual a torch.optim.Adam:", torch.allclose(p, q.detach(), atol=1e-5))
# <<<
# caminos para el diagrama: GD (eta 0,1), momentum (eta 0,02, beta 0,9) y Adam (lo de arriba), 60 pasos
def correr(tipo):
    p = torch.tensor([-4.5, 1.5]); vel = torch.zeros(2); m = torch.zeros(2); s = torch.zeros(2); out = [p.tolist()]; res = []
    for t in range(1, 61):
        g = grad(p)
        if tipo == "gd": p = p - 0.1 * g
        elif tipo == "momentum": vel = 0.9 * vel + g; p = p - 0.02 * vel
        else:
            m = b1 * m + (1 - b1) * g; s = b2 * s + (1 - b2) * g**2
            p = p - eta * (m / (1 - b1**t)) / ((s / (1 - b2**t)).sqrt() + eps)
        out.append(p.tolist())
    return [[round(a, 4), round(b, 4)] for a, b in out], round(perdida(p).item(), 4)
datos = {k: correr(k) for k in ["gd", "momentum", "adam"]}
json.dump(datos, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep10_optim.json"), "w"))
