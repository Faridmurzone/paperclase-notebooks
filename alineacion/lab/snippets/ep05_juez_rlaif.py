import math
CONSTITUCION = {
    "no inventes certezas": ["seguro que", "siempre"],
    "no culpes al usuario": ["es tu culpa"],
}
def criticar(resp, principio):
    return [p for p in CONSTITUCION[principio] if p in resp]
# Juguete: en CAI el juez es un LLM; logp_opcion sería log P("(A)") y log P("(B)").
def logp_opcion(resp, principio):
    return -2.0 * len(criticar(resp, principio))
muestras = [("¿llueve hoy?", "seguro que llueve.", "quizás llueve."),
            ("¿llego a tiempo?", "es tu culpa si no.", "depende del tránsito."),
            ("¿qué hora es?", "son las tres.", "son las tres.")]
# >>>
def juez(a, b, principio):               # P(A mejor), normalizada entre A y B
    la, lb = logp_opcion(a, principio), logp_opcion(b, principio)
    return math.exp(la) / (math.exp(la) + math.exp(lb))

pares = []
for i, (prompt, a, b) in enumerate(muestras):
    principio = list(CONSTITUCION)[i % 2]
    p_a = juez(a, b, principio)                 # etiqueta blanda, no 0/1
    elegida, rechazada = (a, b) if p_a >= 0.5 else (b, a)
    pares.append(dict(prompt=prompt, y_w=elegida, y_l=rechazada, p=p_a))
    print(f"{prompt:17} P(A)={p_a:.2f}  y_w={elegida!r}")
# pares → modelo de preferencias + RL (RLAIF), o directo a DPO
# <<<
