import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
torch.manual_seed(0)


class Capa(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.proj(x)


m = get_peft_model(Capa(), LoraConfig(r=2, lora_alpha=4, target_modules=["proj"], use_dora=True))
q = m.base_model.model.proj
x = torch.randn(5, 8)
# >>>
W0 = q.base_layer.weight                          # d × k, congelada
A, B = q.lora_A["default"].weight, q.lora_B["default"].weight
s = q.scaling["default"]
mag = q.lora_magnitude_vector["default"].weight   # d números, entrena
print("magnitud:", tuple(mag.shape))
print("inicio = normas de W0:",
      torch.allclose(mag, W0.norm(dim=1)))
with torch.no_grad():                             # como tras entrenar
    mag.mul_(1.5)
    B.normal_()

V = W0 + s * B @ A                                # la dirección, sin normalizar
W_dora = mag[:, None] * V / V.norm(dim=1, keepdim=True)
print("normas de W_dora = mag:",
      torch.allclose(W_dora.norm(dim=1), mag))
print(f"máx |peft − mía|: {(q(x) - x @ W_dora.T).abs().max():.1e}")
print("entrenables:", sum(p.numel() for p in m.parameters() if p.requires_grad))
# <<<
