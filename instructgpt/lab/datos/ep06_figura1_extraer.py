"""Digitaliza la figura 1 de InstructGPT (arXiv 2203.02155, página 2) desde los vectores del PDF.
No es un snippet: necesita pymupdf (python3 del sistema). Uso: python3 -I ep06_figura1_extraer.py 2203.02155.pdf > ep06_figura1.json
Eje y: ticks 0,2 / 0,4 / 0,6 en y = 196,6 / 151,5 / 106,4 pt. Control: SFT 175B tiene que dar 0,500 (el baseline contra sí mismo).
"v" = centro del marcador; "ic" = media altura de la barra de error (intervalo del 95 %, según el epígrafe)."""
import pymupdf, json, sys
d = pymupdf.open(sys.argv[1]); p = d[1]
y2v = lambda y: 0.2 + (196.6 - y) / 45.1 * 0.2
nombres = {(0.784, 0.333, 0.239): "PPO-ptx", (0.949, 0.702, 0.353): "PPO", (0.49, 0.729, 0.533): "SFT",
           (0.6, 0.8, 1.0): "GPT (prompted)", (0.443, 0.757, 0.82): "GPT"}
xs = {186.6: "1.3B", 242.8: "6B", 366.9: "175B"}
out = {}
for x in p.get_drawings():
    c = x.get("color"); r = x["rect"]
    if not c or r.y1 > 224 or r.x0 > 376: continue
    k = nombres.get(tuple(round(v, 3) for v in c))
    if not k: continue
    cx = round((r.x0 + r.x1) / 2, 1)
    if cx not in xs: continue
    o = out.setdefault(k, {}).setdefault(xs[cx], {})
    if x["items"][0][0] == "c":
        o["v"] = round(y2v((r.y0 + r.y1) / 2), 3)
    elif x["items"][0][0] == "l" and len(x["items"]) == 1:
        o["ic"] = round((r.y1 - r.y0) / 2 / 45.1 * 0.2, 3)
print(json.dumps(out, indent=1))
