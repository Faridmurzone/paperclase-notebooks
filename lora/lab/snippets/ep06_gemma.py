import torch
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
cfg_gemma = AutoConfig.from_pretrained("google/gemma-2-2b")
# >>>
for nombre, objetivo in [("q y v", ["q_proj", "v_proj"]),
                         ("all-linear", "all-linear")]:
    with torch.device("meta"):                # sin pesos: solo las formas
        base = AutoModelForCausalLM.from_config(cfg_gemma)
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=objetivo)
    m = get_peft_model(base, cfg)
    capas = [n for n, x in m.named_modules() if hasattr(x, "lora_A")]
    entr, total = m.get_nb_trainable_parameters()
    print(f"{nombre + ':':11} {len(capas)} capas,",
          f"{entr:,} ({entr / total:.3%})")

q = m.base_model.model.model.layers[12].self_attn.q_proj
print("capa 12, q_proj: A", tuple(q.lora_A["default"].weight.shape),
      "B", tuple(q.lora_B["default"].weight.shape))
print(f"total: {total:,}")
# <<<
