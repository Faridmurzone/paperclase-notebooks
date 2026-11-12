import importlib.util, os, torch
_r = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "comun", "igpt_juguete.py")
_s = importlib.util.spec_from_file_location("igpt_juguete", _r); j = importlib.util.module_from_spec(_s); _s.loader.exec_module(j)
sft = j.cargar("ep09_sft.pt").eval()
rm, _ = j.cargar_rm("ep09_rm_fuerte.pt")
torch.manual_seed(4)
x = j.prompts(1000, torch.Generator().manual_seed(11))      # los mismos prompts de test del episodio 9
def win_rate(a, b):
    return ((a > b).float() + 0.5 * (a == b).float()).mean().item()
# >>>
politicas = {"PPO-ptx (ep. 9)": j.cargar("ep09_pol_fuerte_0.02.pt"),
             "DPO, γ = 0,1":    j.cargar("ep10_dpo_0.1.pt")}
with torch.no_grad():
    y_sft = j.muestrear(sft, x)
    y = {n: j.muestrear(p.eval(), x) for n, p in politicas.items()}
    for n in politicas:
        print(f"{n:16} vs SFT: juez RM {win_rate(rm(y[n]), rm(y_sft)):.2f}"
              f"  juez oro {win_rate(j.oro(y[n]), j.oro(y_sft)):.2f}")
    ppo, dpo = y.values()
    print(f"DPO vs PPO-ptx, juez oro: {win_rate(j.oro(dpo), j.oro(ppo)):.2f}")
# <<<
