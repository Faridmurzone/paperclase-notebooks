import copy, importlib.util, os, torch
_r = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "comun", "igpt_juguete.py")
_s = importlib.util.spec_from_file_location("igpt_juguete", _r); j = importlib.util.module_from_spec(_s); _s.loader.exec_module(j)
from torch.nn.functional import cross_entropy
base = j.modelo(0)                                         # el mismo preentrenado de ep07: contar ↑
opt = torch.optim.AdamW(base.parameters(), lr=3e-3)
for paso in range(400):
    loss = j.nll_pre(base, j.batch_pre(128)); loss.backward(); opt.step(); opt.zero_grad()
torch.manual_seed(1)
# >>>
sft = copy.deepcopy(base)                       # paso 1: SFT, ahora más corto
opt = torch.optim.AdamW(sft.parameters(), lr=1e-3)
for paso in range(25):
    x = j.demostraciones(32)                    # INST a a−s → la cuenta para atrás
    lg = sft(input_ids=x).logits[:, j.P - 1:-1]
    loss = cross_entropy(lg.reshape(-1, j.V), x[:, j.P:].reshape(-1))
    loss = loss + 0.1 * j.nll_pre(sft, j.batch_pre(32))
    loss.backward(); opt.step(); opt.zero_grad()

y = j.muestrear(sft, j.prompts(512))
print(f"SFT: oro {j.oro(y).mean():.2f} de 6, capacidad {j.capacidad(sft):.3f}")
print("una respuesta:", y[0].tolist())
# <<<
j.guardar(sft, "ep09_sft.pt")
