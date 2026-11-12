import copy, importlib.util, os, torch
_r = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "comun", "igpt_juguete.py")
_s = importlib.util.spec_from_file_location("igpt_juguete", _r); j = importlib.util.module_from_spec(_s); _s.loader.exec_module(j)
from torch.nn.functional import cross_entropy
base = j.modelo(0)
opt = torch.optim.AdamW(base.parameters(), lr=3e-3)
for paso in range(400):                                    # «preentrenamiento»: contar para adelante
    loss = j.nll_pre(base, j.batch_pre(128)); loss.backward(); opt.step(); opt.zero_grad()
torch.manual_seed(1)
# >>>
print(f"preentrenado: capacidad {j.capacidad(base):.3f}")    # benchmark: contar
sft = copy.deepcopy(base)
opt = torch.optim.AdamW(sft.parameters(), lr=1e-3)
for paso in range(60):
    x = j.demostraciones(32)                    # [INST, a, a−s] + contar para atrás
    lg = sft(input_ids=x).logits[:, j.P - 1:-1]
    loss = cross_entropy(lg.reshape(-1, j.V), x[:, j.P:].reshape(-1))
    loss = loss + 0.1 * j.nll_pre(sft, j.batch_pre(32))   # 10 % de preentrenamiento
    loss.backward(); opt.step(); opt.zero_grad()

y = j.muestrear(sft, j.prompts(512))
print(f"SFT: oro {j.oro(y).mean():.2f} de 6, capacidad {j.capacidad(sft):.3f}")
# <<<
j.guardar(sft, "ep07_sft.pt")
