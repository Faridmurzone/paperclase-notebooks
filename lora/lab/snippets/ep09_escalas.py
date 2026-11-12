import math
import torch.nn as nn
from peft import LoraConfig, get_peft_model


class Capa(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 64, bias=False)


# >>>
alpha = 8
print("   r   LoRA α/r   rsLoRA α/√r")
for r in [4, 16, 64]:
    esc = []
    for rs in [False, True]:
        cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=["proj"], use_rslora=rs)
        capa = get_peft_model(Capa(), cfg).base_model.model.proj
        esc.append(capa.scaling["default"])
    print(f"{r:4} {esc[0]:9.3f} {esc[1]:11.3f}")
# <<<
