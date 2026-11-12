import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", dtype=torch.float32)
ids = tok("The food was awful and the waiter was", return_tensors="pt")
r_hat = torch.randn(model.config.hidden_size); r_hat /= r_hat.norm()   # una dirección cualquiera
# >>>
capa = model.model.layers[12]
W = capa.mlp.down_proj.weight                         # [2304, 9216]
W.copy_((torch.eye(2304) - torch.outer(r_hat, r_hat)) @ W)   # ortogonalizar

proy = {}
capa.mlp.register_forward_hook(
    lambda m, i, o: proy.update(mlp=(o[0, -1] @ r_hat).item()))
capa.post_feedforward_layernorm.register_forward_hook(
    lambda m, i, o: proy.update(post=(o[0, -1] @ r_hat).item()))
model(**ids)
print(f"MLP·r̂            = {proy['mlp']:.1e}")       # cero: W' no escribe r̂
print(f"post-norm(MLP)·r̂ = {proy['post']:.1e}")      # va al residual stream
print(type(capa.post_feedforward_layernorm).__name__)
# <<<
