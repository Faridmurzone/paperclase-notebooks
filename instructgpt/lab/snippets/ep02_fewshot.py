# Datos sintéticos y neutros. El procedimiento es el del paper (apéndice A.1).
# >>>
instr = "Dá el sentimiento de la frase."
pares = [("Me gustó el libro.", "Positivo"),     # K pares
         ("El tren llegó tarde.", "Negativo"),        # pregunta/respuesta
         ("Qué lindo día.", "Positivo"),
         ("Perdí el paraguas.", "Negativo")]

ejemplos = []
for i, (q, r) in enumerate(pares):
    otros = [p for j, p in enumerate(pares) if j != i]   # los K−1
    prompt = instr + "\n"
    prompt += "".join(f"{a} → {b}\n" for a, b in otros)  # en contexto
    prompt += q + " →"
    ejemplos.append((prompt, r))                         # objetivo: r

print("K =", len(pares), "pares →", len(ejemplos), "ejemplos")
prompt, r = ejemplos[2]
print(prompt)
print("objetivo:", r)
# <<<
