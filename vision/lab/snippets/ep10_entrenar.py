import os, sys, random
import torch
import torch.nn.functional as F
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from _clip import datos, EncImagen, EncTexto, batch_un_digito_por_clase
torch.manual_seed(0)
Xtr, ytr, _, _ = datos()
enc_i, enc_t = EncImagen(), EncTexto()
rng = random.Random(0)


def infonce(I, T, log_escala):
    logits = log_escala.exp() * I @ T.T
    obj = torch.arange(len(I))
    return (F.cross_entropy(logits, obj) + F.cross_entropy(logits.T, obj)) / 2
# >>>
log_escala = torch.nn.Parameter(torch.tensor(1 / 0.07).log())
params = [*enc_i.parameters(), *enc_t.parameters(), log_escala]
opt = torch.optim.Adam(params, lr=3e-3)
for paso in range(1, 1501):
    imgs, textos = batch_un_digito_por_clase(Xtr, ytr, rng)
    perd = infonce(enc_i(imgs), enc_t(textos), log_escala)
    opt.zero_grad(); perd.backward(); opt.step()
    with torch.no_grad():                 # CLIP: 1/τ nunca pasa de 100
        log_escala.clamp_(max=torch.tensor(100.).log())
    if paso in (1, 100, 500, 1500):
        print(f"paso {paso:4}  pérdida {perd.item():.3f}"
              f"  1/τ = {log_escala.exp().item():.1f}")
# <<<
torch.save({"img": enc_i.state_dict(), "txt": enc_t.state_dict(), "log_escala": log_escala.detach()},
           os.path.join(AQUI, "..", "datos", "clip_ep10.pt"))
