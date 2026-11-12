import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
# >>>
logits = torch.tensor([2.0, 1.5, 1.4, 0.2])      # 1 token, 4 expertos
sesgo = torch.tensor([-0.6, 0.0, 0.3, 0.0])       # e0 sobrecargado, e2 ocioso
probs = logits.softmax(-1)                        # puntaje original

for b in [torch.zeros(4), sesgo]:
    idx = (logits + b).topk(2).indices            # ELEGIR con s + b
    pesos = probs[idx]                            # PESAR con s, sin b
    print("con b:" if b.any() else "sin b:", idx.tolist(), "→",
          [round(p, 3) for p in pesos.tolist()])
print("probs originales:")
print([round(p, 3) for p in probs.tolist()])
# <<<
