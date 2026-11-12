import torch
from transformers import LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, get_peft_model
torch.manual_seed(0)
cfg = LlamaConfig(vocab_size=32, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=4)
base = LlamaForCausalLM(cfg)
# >>>
conf = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base, conf)                   # congela W, agrega A y B
model.print_trainable_parameters()

capa = model.base_model.model.model.layers[0].self_attn.q_proj
W = capa.base_layer.weight                           # congelada
A = capa.lora_A["default"].weight
B = capa.lora_B["default"].weight
print("W", tuple(W.shape), "A", tuple(A.shape), "B", tuple(B.shape))
print("B arranca en cero:", bool((B == 0).all()))

with torch.no_grad():
    B.normal_()                                      # simulamos que ya entrenó
    x = torch.randn(3, 64)
    a_mano = x @ (W + (8 / 4) * B @ A).T             # W' = W + (α/r)·BA
    print("igual a peft:", torch.allclose(capa(x), a_mano, atol=1e-5))
# <<<
