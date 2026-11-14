import os, sys, random
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _clip import datos, EncImagen, EncTexto, batch_un_digito_por_clase
import torch.nn.functional as F
torch.manual_seed(0)
Xtr, ytr, _, _ = datos()
imgs, textos = batch_un_digito_por_clase(Xtr, ytr, random.Random(0))
enc_i, enc_t = EncImagen(), EncTexto()
# >>>
def infonce(img, txt, log_escala):
    I = enc_i(img)                     # N × 32, cada fila con norma 1
    T = enc_t(txt)                     # N × 32
    logits = log_escala.exp() * I @ T.T   # N × N: cosenos / τ
    obj = torch.arange(len(I))         # el par correcto: la diagonal
    perd_i = F.cross_entropy(logits, obj)    # cada imagen elige texto
    perd_t = F.cross_entropy(logits.T, obj)  # cada texto elige imagen
    return (perd_i + perd_t) / 2, logits

log_escala = torch.tensor(1 / 0.07).log()    # τ = 0,07 como en CLIP
perd, logits = infonce(imgs, textos, log_escala)
print("logits:", tuple(logits.shape), "escala 1/τ =",
      round(log_escala.exp().item(), 1))
print("pérdida al azar:", round(perd.item(), 3),
      " ln 10 =", round(torch.tensor(10.).log().item(), 3))
# <<<
