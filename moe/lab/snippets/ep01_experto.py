import glob
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
carpeta = snapshot_download("ibm-granite/granite-3.1-1b-a400m-base", local_files_only=True)
torch.manual_seed(0)
# >>>
capa = "model.layers.0.block_sparse_moe"
for archivo in glob.glob(f"{carpeta}/*.safetensors"):
    with safe_open(archivo, "pt") as f:
        if f"{capa}.input_linear.weight" in f.keys():
            W_in = f.get_slice(f"{capa}.input_linear.weight")[0]    # experto 0
            W_out = f.get_slice(f"{capa}.output_linear.weight")[0]
print("W_in", tuple(W_in.shape), "W_out", tuple(W_out.shape))
x = torch.randn(1024, dtype=W_in.dtype)        # un token
gate, up = (W_in @ x).chunk(2)                 # 1024 → 512 y 512
h = F.silu(gate) * up                          # compuerta: 512
y = W_out @ h                                  # 512 → 1024
print("x", tuple(x.shape), "→ h", tuple(h.shape), "→ y", tuple(y.shape))
n = W_in.numel() + W_out.numel()
print(f"un experto: {n:,}")
print(f"× 32 × 24 capas = {n * 32 * 24:,}")
# <<<
