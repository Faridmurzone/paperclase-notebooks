import glob, json, os, torch
from safetensors import safe_open
d = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/*/"))[0]
indice = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
def cargar(nombre):
    with safe_open(d + indice[nombre], "pt") as f:
        return f.get_tensor(nombre)
# >>>
nombre = "model.layers.12.self_attn.q_proj.weight"
W = cargar(nombre).float()                  # una matriz real de Gemma
m, n = W.shape
print("W:", m, "x", n, "=", f"{m * n:,}", "números")

S = torch.linalg.svdvals(W)                 # 2048 valores singulares
print("sigma_1 =", round(S[0].item(), 2), " sigma_2048 =", round(S[-1].item(), 3))

energia = S**2 / (S**2).sum()
for r in [8, 64, 256, 1024]:
    error = energia[r:].sum().sqrt()        # lo que tiramos
    print(f"r={r:5}: error {error:.0%}  params {r * (m + n):>9,}")
# <<<
# curva error vs rango (Gemma y una matriz al azar de la misma forma) para el diagrama
torch.manual_seed(0)
R = torch.randn(m, n) * W.std()
SR = torch.linalg.svdvals(R); eR = SR**2 / (SR**2).sum()
rangos = [0, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048]
bandas = [(0, 8), (8, 64), (64, 256), (256, 1024), (1024, 2048)]
json.dump({"rangos": rangos,
           "gemma": [round(energia[r:].sum().sqrt().item(), 4) for r in rangos],
           "azar": [round(eR[r:].sum().sqrt().item(), 4) for r in rangos],
           "bandas": [round(energia[a:b].sum().item(), 4) for a, b in bandas]},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep06_curva.json"), "w"))
