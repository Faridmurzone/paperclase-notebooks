import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
torch.manual_seed(0)


class Bloque(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)

    def forward(self, x):
        h = self.q_proj(x) + self.v_proj(x)
        return self.down(torch.relu(self.up(h)))


class LoRALinear(nn.Module):                  # la del episodio cinco
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.escala = alpha / r

    def forward(self, x):
        return self.base(x) + self.escala * (x @ self.A.T @ self.B.T)


cfg = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["q_proj", "v_proj"])
modelo = get_peft_model(Bloque(), cfg)
q = modelo.base_model.model.q_proj
x = torch.randn(5, 8)                         # 5 tokens de d = 8
# >>>
with torch.no_grad():                         # B ≠ 0, como tras entrenar
    q.lora_B["default"].weight.normal_()

mia = LoRALinear(q.base_layer, r=2, alpha=4)
with torch.no_grad():                         # copiamos A y B de peft
    mia.A.copy_(q.lora_A["default"].weight)
    mia.B.copy_(q.lora_B["default"].weight)

y_peft, y_mia = q(x), mia(x)
print("salida:", tuple(y_peft.shape))
print("máx |peft − mía|:", (y_peft - y_mia).abs().max().item())
dW = q.get_delta_weight("default")            # (α/r)·B·A
print("ΔW:", tuple(dW.shape), "rango", torch.linalg.matrix_rank(dW).item())
print("igual a escala·B·A:", torch.allclose(dW, mia.escala * mia.B @ mia.A))
# <<<
