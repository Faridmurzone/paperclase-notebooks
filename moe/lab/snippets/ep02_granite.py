import glob
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeTopKRouter
REPO = "ibm-granite/granite-3.1-1b-a400m-base"
carpeta = snapshot_download(REPO, local_files_only=True)
torch.manual_seed(0)
torch.set_printoptions(precision=2)
# >>>
nombre = "model.layers.0.block_sparse_moe.router.layer.weight"
for archivo in glob.glob(f"{carpeta}/*.safetensors"):
    with safe_open(archivo, "pt") as f:
        if nombre in f.keys():
            W_r = f.get_tensor(nombre).float()       # E × d = 32 × 1024
router = GraniteMoeTopKRouter(AutoConfig.from_pretrained(REPO))
router.weight.data = W_r                             # el router real, capa 0
x = torch.randn(1, 1024)                             # vector al azar
with torch.no_grad():
    logits, g, idx = router(x)
p = logits.softmax(-1)[0, idx[0]]                    # softmax de los 32
print("W_r:", tuple(W_r.shape), "| elegidos:", idx[0].tolist())
print("g:", [round(v, 2) for v in g[0].tolist()])
print("suma:", round(g.sum().item(), 3),
      "| igual a renormalizar:", torch.allclose(g[0], p / p.sum()))
# <<<
