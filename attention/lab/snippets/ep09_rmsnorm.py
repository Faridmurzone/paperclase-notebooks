import torch
import torch.nn as nn
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
# >>>
x = torch.tensor([2., 4., 6., 8.])                 # el token del episodio 6
rms = torch.sqrt((x ** 2).mean() + 1e-6)          # raíz de la media de x²
print(f"RMS = √30 = {rms:.3f} → x / RMS =", [round(v, 3) for v in (x / rms).tolist()])

norma = Gemma2RMSNorm(4)                           # peso w = 0 al inicio
print("igual a Gemma2RMSNorm:", torch.allclose(norma(x), x / rms))
with torch.no_grad():
    norma.weight.fill_(1.0)                        # Gemma escala por (1 + w)
print("con w = 1:", [round(v, 3) for v in norma(x).tolist()])

c = x - x.mean()                                   # un vector con media 0
print("media 0 → RMSNorm == LayerNorm:",
      torch.allclose(Gemma2RMSNorm(4, eps=1e-5)(c), nn.LayerNorm(4)(c), atol=1e-5))
# <<<
