import importlib.util, os, torch
_r = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "comun", "igpt_juguete.py")
_s = importlib.util.spec_from_file_location("igpt_juguete", _r); j = importlib.util.module_from_spec(_s); _s.loader.exec_module(j)
sft = j.cargar("ep09_sft.pt").eval()
casos = [("RM fuerte, β=0,02", "fuerte", 0.02), ("RM débil, β=0", "débil", 0.0), ("RM débil, β=0,1", "débil", 0.1)]
torch.manual_seed(4)
x = j.prompts(1000, torch.Generator().manual_seed(11))      # prompts de test, nuevos
# >>>
def win_rate(a, b):                         # 1 si gana A, ½ si empatan, 0 si pierde
    return ((a > b).float() + 0.5 * (a == b).float()).mean().item()

with torch.no_grad():
    y_sft = j.muestrear(sft, x)             # el baseline responde los mismos prompts
    for nombre, rm_nombre, beta in casos:
        pol = j.cargar(f"ep09_pol_{rm_nombre}_{beta}.pt").eval()
        rm, _ = j.cargar_rm(f"ep09_rm_{rm_nombre}.pt")
        y_pol = j.muestrear(pol, x)
        juez_rm = win_rate(rm(y_pol), rm(y_sft))       # el RM como juez
        juez_oro = win_rate(j.oro(y_pol), j.oro(y_sft))  # el gusto real
        print(f"{nombre:18} juez RM {juez_rm:.2f}   juez oro {juez_oro:.2f}")
# <<<
