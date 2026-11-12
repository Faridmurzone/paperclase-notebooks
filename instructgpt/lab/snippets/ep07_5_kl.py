import importlib.util, os, torch
_r = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "comun", "igpt_juguete.py")
_s = importlib.util.spec_from_file_location("igpt_juguete", _r); j = importlib.util.module_from_spec(_s); _s.loader.exec_module(j)
sft = j.cargar("ep07_sft.pt")
pols = {n: j.cargar(f"ep07_{n}.pt") for n in ["PPO", "PPO-ptx", "β alto"]}
torch.manual_seed(2)
# >>>
@torch.no_grad()
def kl_por_token(pol, x):                  # KL(π‖π_SFT) del próximo token, promediada
    lp, lq = pol(input_ids=x).logits.log_softmax(-1), sft(input_ids=x).logits.log_softmax(-1)
    return (lp.exp() * (lp - lq)).sum(-1).mean().item()

for nombre, pol in pols.items():
    x_rl = j.muestrear(pol, j.prompts(256))          # donde PPO mide el KL
    x_pre = j.batch_pre(256)                         # donde vive el benchmark
    print(f"{nombre:8} KL en prompts RL {kl_por_token(pol, x_rl):.3f}"
          f"   en contar {kl_por_token(pol, x_pre):.3f}")
# <<<
