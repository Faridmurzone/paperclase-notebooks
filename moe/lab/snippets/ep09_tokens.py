import os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _granite import cargar, TEXTOS, capturar_ruteo
import torch
tok, model = cargar()
elegidos, hs = capturar_ruteo(model)
toks = []
for tipo, textos in TEXTOS.items():
    for t in textos:
        ids = tok(t, return_tensors="pt")
        model(**ids)
        toks += tok.convert_ids_to_tokens(ids.input_ids[0])
# >>>
idx = torch.cat(elegidos[12])                 # capa 12: [T, 8]
por_experto = collections.defaultdict(collections.Counter)
for token, fila in zip(toks, idx.tolist()):
    for e in fila:
        por_experto[e][token.replace("Ġ", "_")] += 1

def es_digito(t): return t.strip("_").isdigit()
for e in [10, 27, 23]:
    c = por_experto[e]
    dig = sum(n for t, n in c.items() if es_digito(t)) / sum(c.values())
    top = " ".join(t for t, _ in c.most_common(5))
    print(f"e{e}: {sum(c.values())} tokens, {dig:.0%} dígitos")
    print("   ", top)
# <<<
