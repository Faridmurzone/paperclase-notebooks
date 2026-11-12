import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from _transformer import Transformer, datos, greedy
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep10_transformer.pt")
m = Transformer(); m.load_state_dict(torch.load(CKPT)); m.eval()
# >>>
for largo in (6, 8, 10, 12, 14):                 # entrenado solo con 10
    x, y, _ = datos(1000, torch.Generator().manual_seed(5), largo=largo)
    pred = greedy(m, x)
    print(f"largo {largo:2}: secuencias {(pred == y).all(1).float().mean():6.1%}"
          f" · dígitos {(pred == y).float().mean():.1%}")
# <<<
