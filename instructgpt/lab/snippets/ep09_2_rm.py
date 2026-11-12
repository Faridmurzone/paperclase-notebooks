import importlib.util, itertools, os, torch
import torch.nn.functional as F
_r = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "comun", "igpt_juguete.py")
_s = importlib.util.spec_from_file_location("igpt_juguete", _r); j = importlib.util.module_from_spec(_s); _s.loader.exec_module(j)
sft = j.cargar("ep09_sft.pt").eval()
K = 4
PARES = list(itertools.combinations(range(K), 2))          # (0,1) (0,2) … (2,3): 6 pares
gumbel = torch.distributions.Gumbel(0.0, 1.0)
# >>>
def comparar(n):                     # K respuestas del SFT, ordenadas
    x = j.prompts(n).repeat_interleave(K, 0)
    y = j.muestrear(sft, x).view(n, K, -1)
    gusto = j.oro(y.flatten(0, 1)).view(n, K)        # quien etiqueta: oro
    gusto = gusto + gumbel.sample((n, K))            # … más ruido
    orden = gusto.argsort(-1, descending=True)        # de la mejor a la peor
    return y.gather(1, orden[..., None].expand_as(y))

def entrenar_rm(pasos):
    rm = j.RM(); opt = torch.optim.AdamW(rm.parameters(), lr=1e-3)
    for paso in range(pasos):
        r = rm(comparar(64).flatten(0, 1)).view(64, K)  # un forward por respuesta
        pares = [F.logsigmoid(r[:, a] - r[:, b]) for a, b in PARES]
        loss = -torch.stack(pares).mean()               # los 6 pares
        loss.backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        sesgo = rm(j.demostraciones(512)).mean().item()   # las demos valen 0
    return rm, sesgo
# <<<
@torch.no_grad()
def acierto(rm):                                            # pares nuevos donde el oro distingue
    g = torch.Generator().manual_seed(7)
    x = j.prompts(1000, g)
    torch.manual_seed(7)
    a, b = j.muestrear(sft, x), j.muestrear(sft, x)
    oa, ob = j.oro(a), j.oro(b)
    m = oa != ob
    return ((rm(a) > rm(b)) == (oa > ob))[m].float().mean().item()
for nombre, pasos in [("fuerte", 300), ("débil", 60)]:
    torch.manual_seed(2)
    rm, sesgo = entrenar_rm(pasos)
    print(f"RM {nombre} ({pasos} pasos): acierto {acierto(rm):.3f}  sesgo {sesgo:+.2f}")
    j.guardar_rm(rm, sesgo, f"ep09_rm_{nombre}.pt")
