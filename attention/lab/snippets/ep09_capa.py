import torch
from transformers import AutoConfig
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
cfg = AutoConfig.from_pretrained("google/gemma-2-2b")
# >>>
with torch.device("meta"):                       # solo formas, sin memoria
    capa = Gemma2DecoderLayer(cfg, layer_idx=0)

for nombre, m in capa.named_modules():
    if isinstance(m, torch.nn.Linear):
        print(f"{nombre:16} {tuple(m.weight.shape)}")
normas = [n.replace("_layernorm", "") for n, _ in capa.named_children()
          if "norm" in n]
print(len(normas), "×", type(capa.input_layernorm).__name__,
      tuple(capa.input_layernorm.weight.shape), ":")
print(" ", *normas)
# <<<
