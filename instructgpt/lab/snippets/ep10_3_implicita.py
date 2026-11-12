import importlib.util, os, torch
_r = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "comun", "igpt_juguete.py")
_s = importlib.util.spec_from_file_location("igpt_juguete", _r); j = importlib.util.module_from_spec(_s); _s.loader.exec_module(j)
sft = j.cargar("ep09_sft.pt").eval()
dpo = j.cargar("ep10_dpo_0.1.pt").eval()
rm, sesgo = j.cargar_rm("ep09_rm_fuerte.pt")
g = torch.Generator().manual_seed(7)
x = j.prompts(1000, g)
torch.manual_seed(7)
a, b = j.muestrear(sft, x), j.muestrear(sft, x)               # los mismos pares nuevos del episodio 9
# >>>
beta = 0.5
@torch.no_grad()
def r_implicita(y):                         # β · log π_DPO(y|x) / π_SFT(y|x)
    return beta * (j.logprobs(dpo, y) - j.logprobs(sft, y)).sum(-1)

oa, ob = j.oro(a), j.oro(b)
distintos = oa != ob                        # pares donde el oro elige
for nombre, r in [("RM explícito", rm), ("r implícita DPO", r_implicita)]:
    with torch.no_grad():
        acierto = ((r(a) > r(b)) == (oa > ob))[distintos].float().mean()
    print(f"{nombre:16} ordena bien {acierto:.3f} de los pares")
# <<<
