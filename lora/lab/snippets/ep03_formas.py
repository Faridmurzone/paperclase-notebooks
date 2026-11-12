import glob, json, os, torch
from safetensors import safe_open
d = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/*/"))[0]
indice = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
def cargar(nombre):
    with safe_open(d + indice[nombre], "pt") as f:
        return f.get_tensor(nombre)
torch.manual_seed(0)
# >>>
W0 = cargar("model.layers.12.self_attn.q_proj.weight")   # real, congelada
dout, k = W0.shape                                       # 2048 × 2304
r, alfa = 8, 16
A = torch.randn(r, k) / k**0.5                           # r × k, al azar
B = torch.zeros(dout, r)                                 # d × r, en cero
x = torch.randn(4, k)                                    # 4 tokens

h0 = x @ W0.T                                            # sin LoRA
h = x @ W0.T + (alfa / r) * (x @ A.T) @ B.T              # con LoRA
print("W0", tuple(W0.shape), " A", tuple(A.shape), " B", tuple(B.shape))
print("diferencia máxima con B = 0:", (h - h0).abs().max().item())
print(f"multiplicaciones por token: W0 {dout * k:,}"
      f" | LoRA {r * (k + dout):,}")
# <<<
