import random
random.seed(1)
# Juguete: en Constitutional AI, criticar y revisar son llamadas al mismo LLM con
# un prompt que incluye el principio. Acá son reglas, para que corra en segundos.
CONSTITUCION = {
    "no inventes certezas": ["seguro que", "siempre"],
    "no culpes al usuario": ["es tu culpa"],
}
ARREGLO = {"seguro que": "quizás", "siempre": "a veces",
           "es tu culpa": "le pasa a muchos"}

def revisar(resp, problemas):
    for p in problemas:
        resp = resp.replace(p, ARREGLO[p])
    return resp
# >>>
def criticar(resp, principio):           # en CAI: "¿la respuesta viola…?"
    return [p for p in CONSTITUCION[principio] if p in resp]

resp = "seguro que llueve, siempre pasa. es tu culpa."
inicial, datos_sft = resp, []
for ronda in range(3):
    principio = random.choice(list(CONSTITUCION))   # se sortea por ronda
    problemas = criticar(resp, principio)
    print(f"ronda {ronda}: {principio:20} → {problemas}")
    resp = revisar(resp, problemas)                  # en CAI: otro prompt
print("inicial:", inicial)
print("final:  ", resp)
datos_sft.append(("¿llueve hoy?", resp))   # el SFT se entrena con esto
# <<<
