import numpy as np
rng = np.random.default_rng(0)

def hacer_items(n, largo_delata):        # 2 opciones; la correcta suele ser más larga
    items = []
    for _ in range(n):
        correcta = int(rng.integers(2))
        largos = rng.integers(20, 40, size=2)
        if rng.random() < largo_delata:
            largos[correcta] = largos[1 - correcta] + rng.integers(5, 15)
        items.append(dict(verdad=correcta, correcta=correcta,  # verdad: lo que el modelo
                          largos=largos, dificil=rng.random()))  # puede saber
    return items

def modelo(it):                          # sabe el 55 %; si no, elige la más larga
    return it["verdad"] if it["dificil"] < 0.55 else int(np.argmax(it["largos"]))
mas_larga = lambda it: int(np.argmax(it["largos"]))
def permutar(items):                     # cambia solo la etiqueta del corrector
    return [dict(it, correcta=int(rng.integers(2))) for it in items]
items = hacer_items(400, largo_delata=0.9)
parejos = hacer_items(400, largo_delata=0.0)
# >>>
def evaluar(responder, items, B=2000):
    ok = np.array([responder(it) == it["correcta"] for it in items])
    boot = rng.choice(ok, size=(B, len(ok))).mean(1)     # bootstrap
    return ok.mean(), np.percentile(boot, [2.5, 97.5])

def informe(nombre, responder, items):
    m, (lo, hi) = evaluar(responder, items)
    print(f"{nombre:28} {m:.2f}  IC95 [{lo:.2f}, {hi:.2f}]")

informe("modelo", modelo, items)
informe("ctrl −: etiquetas al azar", modelo, permutar(items))
informe("ctrl −: elegir la más larga", mas_larga, items)
informe("modelo, largos parejos", modelo, parejos)
# <<<
