import os, sys, json
import torch
import torch.nn.functional as F
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from _cifar import cargar, normalizar
from _resnet import cargar_red
torch.set_grad_enabled(False)
X, y = cargar("test", n=1)
vit, _ = cargar_red(f"{AQUI}/../datos/ep09_vit_aug.pt")
capa = vit.encoder.layers[-1]                # el último bloque del ViT
z = vit.tokens(normalizar(X))
for c in vit.encoder.layers[:-1]:
    z = c(z)
# >>>
h = capa.norm1(z)                            # (1, 65, 128): CLS + 64 patches
Wq, Wk, Wv = capa.self_attn.in_proj_weight.chunk(3)
bq, bk, bv = capa.self_attn.in_proj_bias.chunk(3)
H, dh = 4, 128 // 4                          # 4 heads de 32
Q = (h @ Wq.T + bq).view(65, H, dh).transpose(0, 1)   # (4, 65, 32)
K = (h @ Wk.T + bk).view(65, H, dh).transpose(0, 1)
A = F.softmax(Q @ K.transpose(1, 2) / dh ** 0.5, dim=-1)  # (4, 65, 65)

_, A_mod = capa.self_attn(h, h, h, average_attn_weights=False)
print("A:", tuple(A.shape), " dif. con PyTorch:",
      f"{(A - A_mod[0]).abs().max().item():.1e}")
cls = A[:, 0, 1:].mean(0).view(8, 8)          # qué patches mira el CLS
print("cada fila suma:", round(A[0, 0].sum().item(), 3))
# <<<
# datos para las figuras del ep09 (se mezclan con los de ep09_posiciones)
ruta = os.path.join(AQUI, "..", "datos", "ep09_fig.json")
fig = json.load(open(ruta)) if os.path.exists(ruta) else {}
logits0 = (Q[0, 0] @ K[0, 1:].T) / dh ** 0.5            # head 0, CLS contra los 64 patches
top = logits0.argsort(descending=True)[:3].tolist() + logits0.argsort()[:3].tolist()
sub = logits0[top]
fig["cls_labels"] = [f"({j // 8},{j % 8})" for j in top]
fig["cls_vlabels"] = [f"v_{{{j // 8},{j % 8}}}" for j in top]
fig["cls_logits"] = [round(v, 2) for v in sub.tolist()]
fig["cls_probs"] = [round(v, 2) for v in F.softmax(sub, -1).tolist()]
fig["cls_mapa"] = [[round(100 * v, 1) for v in fila] for fila in cls.tolist()]
fig["clase"] = int(y[0])
fig["gato"] = [[round(v, 2) for v in fila] for fila in (0.299 * X[0, 0] + 0.587 * X[0, 1] + 0.114 * X[0, 2]).div(255).tolist()]
json.dump(fig, open(ruta, "w"))
