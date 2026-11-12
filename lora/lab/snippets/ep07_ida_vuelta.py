import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


class Capa(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(2304, 2048, bias=False)   # forma de Gemma

    def forward(self, x):
        return self.q_proj(x)


# >>>
for tipo in [torch.float32, torch.bfloat16]:
    torch.manual_seed(0)
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj"])
    m = get_peft_model(Capa().to(tipo), cfg)
    with torch.no_grad():
        m.base_model.model.q_proj.lora_B["default"].weight.normal_(std=0.01)
    W = m.base_model.model.q_proj.base_layer.weight
    W0 = W.detach().clone()

    m.merge_adapter()                          # W ← W0 + ΔW
    m.unmerge_adapter()                        # W ← W − ΔW
    error = (W.float() - W0.float()).abs()
    print(f"{str(tipo):15} máx error {error.max().item():.1e},",
          f"pesos cambiados {(error > 0).float().mean().item():.1%}")

i = error.argmax()                             # el peso que más se movió
print(f"ej. bf16: {W0.flatten()[i].item():.6f} → {W.flatten()[i].item():.6f}")
# <<<
