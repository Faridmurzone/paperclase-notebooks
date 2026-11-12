import json, os
import torch
torch.manual_seed(0)
n, m = 5, 2                                   # 5 features, 2 dimensiones
importancia = 0.8 ** torch.arange(n)
# >>>
def entrenar(p_cero, pasos=4000):
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(m, n) * 0.1)
    b = torch.nn.Parameter(torch.zeros(n))
    opt = torch.optim.Adam([W, b], lr=1e-2)
    for _ in range(pasos):
        x = torch.rand(1024, n) * (torch.rand(1024, n) > p_cero)  # dispersas
        x_hat = torch.relu(x @ W.T @ W + b)                    # comprimir y volver
        loss = (importancia * (x - x_hat) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach()

for p_cero in (0.0, 0.95):
    W = entrenar(p_cero)
    print(f"P(x_i=0)={p_cero}: |W_i| =", [round(v, 2) for v in W.norm(dim=0).tolist()])
ang = torch.atan2(W[1], W[0]).rad2deg().sort().values
print("ángulos entre vecinas:", torch.diff(ang).round().tolist())
# <<<
json.dump(W.T.tolist(), open(os.path.join(os.path.dirname(__file__), "..", "datos", "ep09_super.json"), "w"))
