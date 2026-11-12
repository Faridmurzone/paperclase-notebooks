import torch
import torch.nn.functional as F
torch.manual_seed(0)
r_w = torch.tensor([2.0, 0.5, -1.0])    # recompensa de la elegida, tres pares
r_l = torch.tensor([0.0, 0.5, 1.0])     # recompensa de la rechazada
# >>>
delta = r_w - r_l                                   # Δr = r(x,y_w) − r(x,y_l)
p = torch.sigmoid(delta)                            # P(y_w ≻ y_l)
loss = -F.logsigmoid(delta)                         # −log σ(Δr), por par
print("Δr:  ", [f"{v:+.1f}" for v in delta])
print("P:   ", [f"{v:.3f}" for v in p])
print("loss:", [f"{v:.3f}" for v in loss])

bce = F.binary_cross_entropy_with_logits(delta, torch.ones(3), reduction="none")
print("igual a BCE con etiqueta 1:", torch.allclose(loss, bce))
corrida = -F.logsigmoid((r_w + 10) - (r_l + 10))
print("¿y si sumo 10 a todo?", torch.allclose(loss, corrida))
# <<<
