import torch
import torch.nn as nn
from transformer_lens import HookedRootModule
from transformer_lens.hook_points import HookPoint
torch.manual_seed(0)
# >>>
class Mini(HookedRootModule):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 3)
        self.hook_mid = HookPoint()          # identidad con nombre
        self.setup()                         # registra los nombres

    def forward(self, x):
        return self.hook_mid(self.lin(x)) * 10

m = Mini()
x = torch.randn(1, 3)
print(isinstance(m.hook_mid, nn.Module), m.hook_mid.name)
_, cache = m.run_with_cache(x)
print(list(cache.keys()))
cero = lambda act, hook: act * 0             # firma TL: (act, hook)
print(m.run_with_hooks(x, fwd_hooks=[("hook_mid", cero)]))
# <<<
