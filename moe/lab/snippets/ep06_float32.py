import inspect
from transformers.models.granitemoe import modeling_granitemoe as gm
# >>>
fuente = inspect.getsource(gm.GraniteMoeTopKRouter.forward)
for linea in fuente.splitlines():
    if "F.linear" in linea or "softmax" in linea:
        print(linea.strip().split("  #")[0])
# <<<
