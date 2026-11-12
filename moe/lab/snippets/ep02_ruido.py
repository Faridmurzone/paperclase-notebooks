import torch
import torch.nn.functional as F
torch.manual_seed(0)
x = torch.tensor([1.0, 0.5, -1.0])
W_g = torch.tensor([[1.0, 0.0, 0.5, -1.0], [0.0, 2.0, 0.0, 1.0], [0.5, 0.0, -1.0, 0.0]])
W_noise = torch.full((3, 4), 0.5)
# >>>
def noisy_top_k(x, k=2):
    limpio = x @ W_g                                      # (x · W_g)_i
    ruido = torch.randn(4) * F.softplus(x @ W_noise)      # ε · softplus
    H = limpio + ruido
    corte = H.topk(k).values[-1]
    H = H.masked_fill(H < corte, float("-inf"))           # KeepTopK
    return H.softmax(-1)                                  # G(x)

veces = torch.zeros(4)
for _ in range(1000):
    veces += noisy_top_k(x) > 0
print("sin ruido, top-2:", (x @ W_g).topk(2).indices.tolist())
pct = (veces / 10).round().int().tolist()            # de 1000 a %
print("con ruido, % elegido:", pct)
# <<<
