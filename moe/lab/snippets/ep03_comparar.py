import json
import os
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
# >>>
# entrenados una vez con lab/preparar_ep03.py (3 semillas, 1500 pasos)
r = json.load(open(os.path.join(DATOS, "ep03_resultados.json")))
print(f"{'modelo':<14}{'totales':>8}{'activos':>8}{'MSE test':>10}")
for nombre, v in r.items():
    mse = sum(v["test"]) / len(v["test"])
    print(f"{nombre:<14}{v['params_capa']:>8}{v['activos_capa']:>8}{mse:>10.3f}")
# <<<
