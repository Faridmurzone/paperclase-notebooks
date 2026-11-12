import re
# >>>
fuente = ("La biblioteca abre de lunes a viernes de 9 a 18. "
          "Los sábados abre de 10 a 14. Los domingos está cerrada.")
resumenes = ["Abre de lunes a viernes de 9 a 18 y los sábados de 10 a 14.",
             "Abre todos los días de 9 a 18.",
             "Los domingos está cerrada; los sábados abre de 10 a 14.",
             "Abre de lunes a viernes y tiene wifi gratis."]

vacias = {"y", "de", "a", "la", "los", "el"}             # no cuentan
palabras = lambda t: set(re.findall(r"\w+", t.lower())) - vacias
vocab = palabras(fuente)
for r in resumenes:
    nuevas = palabras(r) - vocab               # lo que no estaba en la entrada
    print(f"{'ALUCINA' if nuevas else 'ok     '} {sorted(nuevas)}")
tasa = sum(bool(palabras(r) - vocab) for r in resumenes) / len(resumenes)
print(f"tasa de alucinación: {tasa:.2f}")
# <<<
