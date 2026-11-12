# Candidatos sintéticos. Los cortes son los del paper (apéndice B.1): 75 % y 6/7.
candidatos = {"A": (0.81, 0.78, 6.4), "B": (0.92, 0.70, 6.1), "C": (0.77, 0.83, 5.6),
              "D": (0.88, 0.79, 6.5), "E": (0.66, 0.74, 6.2)}
# >>>
def pasa(sensible, ranking, demo):
    return [sensible >= 0.75,        # acuerdo marcando contenido sensible
            ranking >= 0.75,         # acuerdo ordenando respuestas
            demo >= 6]               # demostraciones sensibles, de 1 a 7

for nombre, notas in candidatos.items():
    ok = pasa(*notas)
    marca = "".join("✓" if x else "✗" for x in ok)
    s, r, d = notas
    print(f"{nombre} {s:.2f} {r:.2f} {d:.1f} {marca}",
          "entra" if all(ok) else "")
# <<<
