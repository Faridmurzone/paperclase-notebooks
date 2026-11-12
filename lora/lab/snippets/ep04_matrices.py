import glob, json, os
from safetensors import safe_open
d = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--google--gemma-2-2b/snapshots/*/"))[0]
indice = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
def forma(nombre):
    with safe_open(d + indice[nombre], "pt") as f:
        return f.get_slice(nombre).get_shape()
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
# >>>
r = 8
filas = []
for m in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
          "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
    dout, k = forma(f"model.layers.0.{m}.weight")       # W es d × k
    lora = r * (dout + k)                                # B: d × r, A: r × k
    filas.append((m.split(".")[1], dout, k, dout * k, lora))
    print(f"{m.split('.')[1]:9} {dout:5}×{k:<5} {dout * k:>11,}"
          f"  LoRA {lora:>7,} ({lora / (dout * k):.2%})")
# <<<
json.dump(filas, open(os.path.join(DATOS, "ep04_matrices.json"), "w"))
