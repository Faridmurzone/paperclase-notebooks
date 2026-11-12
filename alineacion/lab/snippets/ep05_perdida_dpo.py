import torch
import torch.nn.functional as F
# log-probs (suma sobre tokens) de 3 pares, política y referencia
lp_w  = torch.tensor([-12.0, -20.0, -15.0])
lp_l  = torch.tensor([-14.0, -18.0, -15.0])
ref_w = torch.tensor([-13.0, -20.0, -15.0])
ref_l = torch.tensor([-13.0, -19.0, -15.0])
# >>>
def perdida_dpo(lp_w, lp_l, ref_w, ref_l, beta=0.1):
    r_w = beta * (lp_w - ref_w)        # recompensa implícita de la elegida
    r_l = beta * (lp_l - ref_l)        # recompensa implícita de la rechazada
    margen = r_w - r_l
    return -F.logsigmoid(margen), r_w, r_l

loss, r_w, r_l = perdida_dpo(lp_w, lp_l, ref_w, ref_l)
for i in range(3):
    print(f"par {i}: r_w {r_w[i]:+.2f}  r_l {r_l[i]:+.2f}  "
          f"loss {loss[i]:.3f}")
print(f"log 2 = {torch.log(torch.tensor(2.)):.3f}")
# <<<
