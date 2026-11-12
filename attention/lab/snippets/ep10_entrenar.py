import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from _transformer import Transformer, datos, forma   # el de las escenas anteriores
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep10_transformer.pt")
torch.manual_seed(0)
# >>>
m = Transformer(d=32, h=4, N=2, d_ff=128, p=0.1)
opt = torch.optim.Adam(m.parameters(), lr=2e-3, betas=(0.9, 0.98), eps=1e-9)
sch = LambdaLR(opt, lambda s: forma(s + 1, warmup=100))       # ecuación (3)
m.train()
for paso in range(1, 1201):
    x, y, y_in = datos(64)                       # ordenar 10 dígitos
    logits = m(x, y_in)                          # [64, 10, 11]
    loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                           label_smoothing=0.1)
    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    if paso % 300 == 0:
        print(f"paso {paso:4}: pérdida {loss.item():.3f}")
torch.save(m.state_dict(), CKPT)
# <<<
