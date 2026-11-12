import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
torch.manual_seed(0)


class Bloque(nn.Module):                      # el juguete del episodio 6
    def __init__(self, d=8):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)

    def forward(self, x):
        h = self.q_proj(x) + self.v_proj(x)
        return self.down(torch.relu(self.up(h)))


cfg = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["q_proj", "v_proj"])
modelo = get_peft_model(Bloque(), cfg)
with torch.no_grad():                          # B ≠ 0, como tras entrenar
    for n, p in modelo.named_parameters():
        if "lora_B" in n:
            p.normal_()
x = torch.randn(5, 8)
# >>>
q = modelo.base_model.model.q_proj
W0 = q.base_layer.weight.detach().clone()
dW = q.get_delta_weight("default")             # (α/r)·B·A
y_antes = modelo(x)

fusionado = modelo.merge_and_unload()          # W ← W0 + ΔW
print("q_proj ahora:", type(fusionado.q_proj).__name__,
      "de", type(fusionado.q_proj).__module__.split(".")[0])
W = fusionado.q_proj.weight
print("W − W0 = ΔW:", torch.allclose(W - W0, dW, atol=1e-6))
print("máx |antes − después|:",
      f"{(y_antes - fusionado(x)).abs().max().item():.1e}")
print("parámetros:", sum(p.numel() for p in fusionado.parameters()))
# <<<
