import importlib.util, os, torch
_r = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "comun", "igpt_juguete.py")
_s = importlib.util.spec_from_file_location("igpt_juguete", _r); j = importlib.util.module_from_spec(_s); _s.loader.exec_module(j)
sft = j.cargar("ep09_sft.pt").eval()
torch.manual_seed(8)
x = j.prompts(2000, torch.Generator().manual_seed(13))
a, b = j.muestrear(sft, x), j.muestrear(sft, x)
# >>>
@torch.no_grad()
def juez(a, b):                     # la IA compara: log-verosimilitud de cada respuesta
    la = j.logprobs(sft, a).sum(-1)       # el propio SFT hace de juez
    lb = j.logprobs(sft, b).sum(-1)
    return torch.sigmoid(la - lb)         # P(A mejor): etiqueta blanda

p = juez(a, b)
p_rec = p.clamp(0.4, 0.6)                 # recorte 40–60 % de CAI
oa, ob = j.oro(a), j.oro(b)
m = oa != ob
print(f"acierto del juez IA:  {((p > 0.5) == (oa > ob))[m].float().mean():.3f}")
print(f"etiquetas casi 0 o 1: {((p < 0.05) | (p > 0.95)).float().mean():.0%}")
print(f"después del recorte:  p entre {p_rec.min():.1f} y {p_rec.max():.1f}")
# <<<
