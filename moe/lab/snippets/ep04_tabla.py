import json
import os
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
# >>>
r = json.load(open(os.path.join(DATOS, "ep04_resultados.json")))
print(f"{'α':<6}{'MSE test':>9}{'muertos':>9}{'máx/medio':>11}")
for alfa, v in r.items():                         # 3 semillas cada uno
    mse = sum(v["test"]) / 3
    muertos = sum(sum(c < 0.01 * sum(cs) for c in cs)
                  for cs in v["carga_final"]) / 3          # < 1 % de carga
    ratio = sum(max(cs) / (sum(cs) / 8) for cs in v["carga_final"]) / 3
    print(f"{alfa:<6}{mse:>9.4f}{muertos:>9.1f}{ratio:>11.2f}")
# <<<
