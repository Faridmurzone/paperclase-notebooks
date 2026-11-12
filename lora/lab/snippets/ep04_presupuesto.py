import json, os
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")
filas = {f[0]: f for f in json.load(open(os.path.join(DATOS, "ep04_matrices.json")))}
N = json.load(open(os.path.join(DATOS, "ep01_conteo.json")))["total"]
# >>>
def entrenables(objetivos, r, capas=26):
    por_capa = sum(r * (filas[m][1] + filas[m][2]) for m in objetivos)
    return capas * por_capa

opciones = {"q, v": ["q_proj", "v_proj"],
            "q, k, v, o": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "todas las lineales": list(filas)}
for nombre, objetivos in opciones.items():
    for r in [8, 64]:
        n = entrenables(objetivos, r)
        print(f"{nombre:18} r={r:2}: {n:>11,} ({n / N:.2%})")
# <<<
