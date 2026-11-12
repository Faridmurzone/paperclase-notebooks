import json, math, os
import torch
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
torch.manual_seed(0)
# >>>
n, capas = 256, 20
x0 = torch.randn(512, n)                    # un batch de 512

def propagar(std_w, act):
    torch.manual_seed(1)                    # mismos pesos al azar
    x, stds = x0, []
    for _ in range(capas):                  # 20 capas sin entrenar
        W = torch.randn(n, n) * std_w
        x = act(x @ W.T)
        stds.append(x.std().item())         # ¿cuánta señal queda?
    return stds

casos = {"tanh 0,01": (0.01, torch.tanh),           # std de los pesos
         "ReLU 1/n": (1 / math.sqrt(n), torch.relu),  # varianza 1/n
         "ReLU 2/n": (math.sqrt(2 / n), torch.relu)}  # varianza 2/n
for nombre, (s, act) in casos.items():
    st = propagar(s, act)
    print(f"{nombre:10}", " ".join(f"{st[i]:.4f}" for i in (0, 4, 9, 19)))
# <<<
json.dump({k: propagar(s, a) for k, (s, a) in casos.items()}, open(os.path.join(DATOS, "ep10_senal.json"), "w"))
