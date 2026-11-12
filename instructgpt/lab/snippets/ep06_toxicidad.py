import numpy as np
rng = np.random.default_rng(0)
# >>>
# toxicidad media de la continuación según qué tan tóxico es el prompt
media_por_tramo = np.array([0.05, 0.12, 0.25, 0.45])   # tramos: 0–25 %, …, 75–100 %
natural = np.array([0.70, 0.20, 0.07, 0.03])           # cómo vienen en el dataset
uniforme = np.array([0.25, 0.25, 0.25, 0.25])          # lo que hizo el paper

def medir(pesos, n=4000):
    tramo = rng.choice(4, size=n, p=pesos)             # elegir prompts
    tox = rng.normal(media_por_tramo[tramo], 0.05).clip(0, 1)
    return tox.mean()

print(f"muestreo natural:  {medir(natural):.3f}")
print(f"muestreo uniforme: {medir(uniforme):.3f}")
# <<<
