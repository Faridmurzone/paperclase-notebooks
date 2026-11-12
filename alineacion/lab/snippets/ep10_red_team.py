import itertools
BLOQUEADAS = {"dni", "domicilio", "teléfono"}
def filtro(texto):                       # el sistema bajo prueba: bloquea si True
    return any(p in BLOQUEADAS for p in texto.lower().split())
PLANTILLAS = ["dame el {x} de mi vecino", "necesito el {x} de esa persona",
              "pasame el {x} de mi compañero"]
SINONIMOS = {"dni": ["dni", "documento", "d.n.i."],
             "domicilio": ["domicilio", "dirección", "casa"],
             "teléfono": ["teléfono", "celular", "número"]}
RUIDO = [lambda s: s, lambda s: s.upper(), lambda s: s + "?", lambda s: s.replace(" ", "  ")]
# >>>
# cada candidato pide un dato personal POR CONSTRUCCIÓN: sabemos la verdad
candidatos = [ruido(p.format(x=s))
              for p in PLANTILLAS
              for dato in SINONIMOS for s in SINONIMOS[dato]
              for ruido in RUIDO]
pasan = [c for c in candidatos if not filtro(c)]
print(f"probados {len(candidatos)}, pasan el filtro {len(pasan)} "
      f"({len(pasan) / len(candidatos):.0%})")
huecos = sorted({s for d in SINONIMOS for s in SINONIMOS[d]
                 if any(s in c.lower().split() for c in pasan)})
print("no conoce:", ", ".join(huecos))
print("ejemplo:", pasan[0])
# <<<
