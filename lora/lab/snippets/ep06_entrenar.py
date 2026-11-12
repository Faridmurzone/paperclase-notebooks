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


cfg = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["q_proj", "v_proj"])
modelo = get_peft_model(Bloque(), cfg)
X, Y = torch.randn(64, 8), torch.randn(64, 8)    # datos sintéticos
# >>>
q = modelo.base_model.model.q_proj
W_antes = q.base_layer.weight.clone()
entrenables = [p for p in modelo.parameters() if p.requires_grad]
opt = torch.optim.AdamW(entrenables, lr=1e-2)

for paso in range(200):
    perdida = ((modelo(X) - Y) ** 2).mean()
    if paso == 0:
        print(f"pérdida inicial: {perdida.item():.3f}")
    opt.zero_grad(); perdida.backward(); opt.step()

print(f"pérdida final:   {perdida.item():.3f}")
print("grad de W:", q.base_layer.weight.grad)
print("W intacta:", torch.equal(W_antes, q.base_layer.weight))
print(f"máx |B| ahora: {q.lora_B['default'].weight.abs().max():.2f}")
# <<<
