from math import comb
from transformers import AutoConfig
# >>>
cfg = AutoConfig.from_pretrained("ibm-granite/granite-3.1-1b-a400m-base")
E, k = cfg.num_local_experts, cfg.num_experts_per_tok
d, h = cfg.hidden_size, cfg.intermediate_size
print(f"E={E}, k={k}, d={d}, h={h}")
print("claves 'shared':", [c for c in cfg.to_dict() if "shared" in c])
por_experto = 3 * d * h                       # gate, up y down
print(f"activos/capa: {k} × {por_experto / 1e6:.2f} M = "
      f"{k * por_experto / 1e6:.1f} M")
print(f"grueso igual: {E // 4} de h={4 * h}, top-{k // 4}")
print(f"C({E},{k}) = {comb(E, k):,} · C(8,2) = {comb(8, 2)}")
# <<<
