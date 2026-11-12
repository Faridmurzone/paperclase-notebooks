import random
from collections import Counter
random.seed(0)
# prompts sintéticos: (usuario, texto). Algunos comparten un comienzo largo.
base = ["Resumí este texto en una línea: ", "Escribí un poema sobre ", "Listá cinco ideas para ",
        "Traducí al inglés: ", "Explicá como a un chico: "]
temas = ["el mar", "la lluvia", "un faro", "el otoño", "los trenes", "un jardín", "la luna", "un puente",
         "el viento", "una plaza", "los ríos", "un volcán"]
prompts = []
for u in "abcdefgh":
    for _ in range(random.randint(2, 9)):
        b = random.choice(base) if u != "c" else base[0]          # el usuario c repite plantilla
        t = random.choice(temas) if u != "c" else "el informe semanal de " + random.choice(temas)
        prompts.append((u, b + t))
random.shuffle(prompts)
TOPE = 3                        # el paper usa 200 por usuario; acá, a escala
LARGO = 40                      # largo del prefijo que se compara
# >>>
vistos, por_usuario, limpios = set(), Counter(), []
for usuario, p in prompts:
    if p[:LARGO] in vistos:                  # mismo comienzo largo: duplicado
        continue
    if por_usuario[usuario] >= TOPE:         # tope de prompts por usuario
        continue
    vistos.add(p[:LARGO])
    por_usuario[usuario] += 1
    limpios.append((usuario, p))

usuarios = sorted(por_usuario)
random.shuffle(usuarios)
test = set(usuarios[:2])                     # se divide por usuario, no por prompt
train = [(u, p) for u, p in limpios if u not in test]
print("prompts:", len(prompts), "→ limpios:", len(limpios))
antes = Counter(u for u, _ in prompts)
print("antes:  ", " ".join(f"{u}{antes[u]}" for u in sorted(antes)))
print("después:", " ".join(f"{u}{por_usuario[u]}" for u in sorted(antes)))
print("test:", sorted(test), "| en ambos:",
      len({u for u, _ in train} & test))
# <<<
import json, os
json.dump({"antes": dict(sorted(Counter(u for u, _ in prompts).items())), "despues": dict(sorted(por_usuario.items())),
           "test": sorted(test)},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep02_filtrar.json"), "w"))
