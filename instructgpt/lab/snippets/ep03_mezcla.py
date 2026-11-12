import torch
torch.manual_seed(0)
demos = [f"demo{i}" for i in range(12_725)]           # prompts de SFT de train (tabla 6)
preentreno = [f"web{i}" for i in range(1_000_000)]    # texto de preentrenamiento (sintético)
# >>>
def batch_mezclado(B=32, frac=0.10):           # init de PPO: 10 % de preentreno
    n_pre = round(B * frac)
    i = torch.randint(len(demos), (B - n_pre,))
    j = torch.randint(len(preentreno), (n_pre,))
    return [demos[k] for k in i] + [preentreno[k] for k in j]

b = batch_mezclado()
n_demo = sum(x.startswith("demo") for x in b)
print(f"{len(b)} ejemplos: {n_demo} demos + {len(b) - n_demo} preentreno")
# <<<
