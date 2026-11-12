import torch
from transformer_lens import TransformerBridge
torch.manual_seed(0)
# >>>
model = TransformerBridge.boot_transformers("google/gemma-2-2b",
                                            dtype=torch.float32, device="cpu")
texto = "The Eiffel Tower is in the city of"
logits, cache = model.run_with_cache(texto)
print(len(cache), "activaciones guardadas")
print(cache["blocks.12.hook_resid_post"].shape)
print([k[10:] for k in cache if k.startswith("blocks.12.hook_")][:5])

paris = model.to_single_token(" Paris")
def borrar_mlp(act, hook):                 # act: [batch, pos, d]
    act[:, -1] = 0
    return act
for capa in (2, 12, 22):
    l = model.run_with_hooks(texto, fwd_hooks=[
        (f"blocks.{capa}.hook_mlp_out", borrar_mlp)])
    print(capa, f"{l[0, -1].softmax(-1)[paris]:.3f}  (sin hook: "
          f"{logits[0, -1].softmax(-1)[paris]:.3f})")
# <<<
