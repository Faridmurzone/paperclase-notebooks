import importlib.util, os, torch
_r = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "comun", "igpt_juguete.py")
_s = importlib.util.spec_from_file_location("igpt_juguete", _r); j = importlib.util.module_from_spec(_s); _s.loader.exec_module(j)
rm, sesgo = j.cargar_rm("ep09_rm_débil.pt")
torch.manual_seed(5)
x = j.prompts(3, torch.Generator().manual_seed(21))
# >>>
for nombre in ["ep09_sft.pt", "ep09_pol_débil_0.0.pt"]:      # antes y después
    pol = j.cargar(nombre).eval()
    with torch.no_grad():
        y = j.muestrear(pol, x, greedy=True)
        puntaje = rm(y) - sesgo
    print(nombre.removesuffix(".pt"))
    for fila, p, o in zip(y.tolist(), puntaje, j.oro(y)):
        print(f"  {fila[1:3]} → {fila[3:]}  RM {p:+.2f}  oro {o:.0f}")
# <<<
