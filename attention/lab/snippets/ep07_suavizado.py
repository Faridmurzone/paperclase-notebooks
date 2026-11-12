import torch
import torch.nn.functional as F
# >>>
V, eps, y = 5, 0.1, torch.tensor([0])            # 5 clases, la correcta es la 0
q = torch.full((V,), eps / V); q[0] += 1 - eps   # objetivo suavizado
print("objetivo:", [round(v, 2) for v in q.tolist()])

modelos = {"seguro":   torch.tensor([[9., 0, 0, 0, 0]]),
           "como q":   q.log().unsqueeze(0),      # predice exactamente q
           "uniforme": torch.zeros(1, V)}
for nombre, logits in modelos.items():
    p = logits.softmax(-1)[0]
    duro = F.cross_entropy(logits, y)
    suave = F.cross_entropy(logits, y, label_smoothing=eps)
    a_mano = -(q * p.log()).sum()
    print(f"{nombre:8} p0={p[0]:.2f}  sin ε {duro:.2f}  con ε {suave:.2f}"
          f"  a mano {a_mano:.2f}")
# <<<
