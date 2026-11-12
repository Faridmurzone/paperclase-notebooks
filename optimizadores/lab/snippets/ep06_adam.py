import json, os
import torch
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")

def ruido(t):                                   # gradiente de la pérdida: ±escala, promedio cero
    return torch.tensor([10.0, 0.01]) * (1 if t % 2 == 0 else -1)

# >>>
eta, lam, b1, b2, eps = 0.01, 0.1, 0.9, 0.999, 1e-8
def adam(decay_aparte, pasos=300):
    th, m, v = torch.tensor([1.0, 1.0]), torch.zeros(2), torch.zeros(2)
    hist = [th.clone()]
    for t in range(1, pasos + 1):
        g = ruido(t)                            # A: escala 10; B: escala 0,01
        if decay_aparte: th = th * (1 - eta * lam)   # AdamW
        else:            g = g + lam * th            # Adam + L2
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
        th = th - eta * mh / (vh.sqrt() + eps)
        hist.append(th.clone())
    return torch.stack(hist)
l2, w = adam(False), adam(True)
for nombre, h in [("Adam + L2", l2), ("AdamW    ", w)]:
    print(nombre, "paso 300 → A:", round(h[-1, 0].item(), 3),
          " B:", round(h[-1, 1].item(), 3))
# <<<
json.dump({"l2": l2.numpy().round(4).tolist(), "w": w.numpy().round(4).tolist()},
          open(os.path.join(DATOS, "ep06_adam.json"), "w"))
