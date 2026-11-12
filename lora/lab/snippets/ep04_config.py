import glob, json, os
from safetensors import safe_open
d = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/*/"))[0]
indice = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
def forma(nombre):
    with safe_open(d + indice[nombre], "pt") as f:
        return f.get_slice(nombre).get_shape()
# >>>
cfg = json.load(open(d + "config.json"))                # la ficha del modelo
for clave in ["hidden_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "intermediate_size"]:
    print(f"{clave:20} {cfg[clave]}")

q = cfg["num_attention_heads"] * cfg["head_dim"]         # salidas de q_proj
kv = cfg["num_key_value_heads"] * cfg["head_dim"]        # salidas de k y v
print("q_proj sale de", q, "| k_proj y v_proj, de", kv)
# <<<
