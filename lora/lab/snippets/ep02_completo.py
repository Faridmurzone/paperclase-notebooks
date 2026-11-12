import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import *
torch.manual_seed(0)
# >>>
red = cargar_base()                                    # la red preentrenada
W0 = [red[i].weight.detach().clone() for i in (0, 2, 4)]

opt = torch.optim.Adam(red.parameters(), lr=1e-3)      # ajuste completo
for paso in range(300):
    i = torch.randint(0, len(Xtr), (64,))              # batch de 64
    perdida = F.cross_entropy(red(espejo(Xtr[i])), ytr[i])
    opt.zero_grad(); perdida.backward(); opt.step()

dW = [red[i].weight.detach() - w0 for i, w0 in zip((0, 2, 4), W0)]
print("en espejo, después:", f"{precision(red, espejo(Xte), yte):.1%}")
for d, w0 in zip(dW, W0):
    print(tuple(d.shape), f"|ΔW|/|W0| = {d.norm() / w0.norm():.2f}")
# <<<
torch.save(dict(W0=W0, dW=dW, red=red.state_dict()), os.path.join(DATOS, "ep02_completo.pt"))
