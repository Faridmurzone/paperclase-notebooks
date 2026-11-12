import json
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.gemma2.modeling_gemma2 import Gemma2MLP
torch.manual_seed(0)
REPO = "google/gemma-2-2b"
cfg = AutoConfig.from_pretrained(REPO)
mapa = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))["weight_map"]


def peso(n):
    with safe_open(hf_hub_download(REPO, mapa[n]), framework="pt") as f:
        return f.get_tensor(n).float()


# >>>
mlp = Gemma2MLP(cfg)                                     # pesos reales, capa 0
for p in ("gate_proj", "up_proj", "down_proj"):
    getattr(mlp, p).weight.data = peso(f"model.layers.0.mlp.{p}.weight")

x = torch.randn(3, cfg.hidden_size)                      # 3 tokens
g = F.gelu(x @ mlp.gate_proj.weight.T, approximate="tanh")   # la compuerta
u = x @ mlp.up_proj.weight.T                             # lo que pasa
a_mano = (g * u) @ mlp.down_proj.weight.T
print("a mano == Gemma2MLP:", torch.allclose(a_mano, mlp(x), atol=1e-4))
print("x", tuple(x.shape), "→ g y u", tuple(g.shape), "→ salida", tuple(a_mano.shape))

n = sum(p.numel() for p in mlp.parameters())
print(f"parámetros: {n / 1e6:.1f} M (el paper: 2.1 M) · 2 × 9216 = {2 * 9216}")
# <<<
