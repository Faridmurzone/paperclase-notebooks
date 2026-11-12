import copy
import json
import os
import statistics
import time
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
torch.manual_seed(0)
torch.set_num_threads(4)
d = 2304
base = nn.Sequential(*[nn.Linear(d, d, bias=False) for _ in range(4)])
x = torch.randn(512, d)                        # 512 tokens


def medir(m, n=20):                            # mediana de 20 forwards, en ms
    with torch.no_grad():
        for _ in range(3):
            m(x)
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            m(x)
            ts.append(time.perf_counter() - t)
    return 1e3 * statistics.median(ts)


# >>>
cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["0", "1", "2", "3"])
con_lora = get_peft_model(copy.deepcopy(base), cfg).eval()
t_base, t_lora = medir(base), medir(con_lora)
t_merge = medir(con_lora.merge_and_unload())

print(f"base sola:      {t_base:5.2f} ms")
print(f"LoRA sin merge: {t_lora:5.2f} ms  (+{t_lora / t_base - 1:.0%})")
print(f"LoRA fusionado: {t_merge:5.2f} ms")
r = 8
print(f"multiplicaciones extra: {r * (d + d) / (d * d):.2%}")
# <<<
json.dump(dict(base=t_base, lora=t_lora, merge=t_merge),
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep07_costo.json"), "w"))
