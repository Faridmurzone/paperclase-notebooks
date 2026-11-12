import glob, json, os, torch
from safetensors import safe_open
from transformers import AutoTokenizer
d = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/*/"))[0]
indice = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
palabras = [" one", " two", " three", " four", " five", " six", " seven", " eight",
            " red", " blue", " green", " yellow", " black", " white", " purple", " orange",
            " Monday", " Tuesday", " Wednesday", " Thursday", " Friday", " Saturday", " Sunday", " weekend"]
ids = [tok.encode(p, add_special_tokens=False)[0] for p in palabras]
k = "model.embed_tokens.weight"
with safe_open(d + indice[k], "pt") as f:
    tabla = f.get_slice(k)
    X = torch.stack([tabla[i:i + 1][0] for i in ids]).float()
# >>>
# X: 24 embeddings de Gemma (8 números, 8 colores, 8 días)
print("X:", tuple(X.shape))

Xc = X - X.mean(dim=0)                      # 1) centrar
U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
var = S**2 / (S**2).sum()                   # varianza por componente
print("varianza PC1, PC2:", [round(v, 2) for v in var[:2].tolist()])

P = Xc @ Vt[:2].T                           # 2) proyectar en 2 dims
for g, nombre in enumerate(["números", "colores", "días"]):
    c = P[8 * g: 8 * g + 8].mean(dim=0)
    print(f"{nombre:8} centro en ({c[0]:+.2f}, {c[1]:+.2f})")
# <<<
json.dump(P.round(decimals=2).tolist(), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep06_pca.json"), "w"))
