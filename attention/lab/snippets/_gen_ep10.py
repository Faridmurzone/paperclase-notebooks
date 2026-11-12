"""Genera los snippets ep10_* copiando textual las clases de _transformer.py (fuente única). Correr: python3 _gen_ep10.py"""
import os
import re
AQUI = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(AQUI, "_transformer.py")).read()
bloque = lambda n: re.search(rf"# ── {n} ──\n(.*?)# ── FIN ──", SRC, re.S).group(1).rstrip() + "\n"
CAB = '''import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from _transformer import {imp}   # el resto del transformer (mismo código)
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep10_transformer.pt")
torch.manual_seed(0)
'''
SNIP = {
 "ep10_atencion": ("V, BOS, datos", bloque("MULTIHEAD"), '''
att, x = MultiHead(d=32, h=4), torch.randn(2, 5, 32)        # batch 2, 5 tokens
causal = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
x2 = x.clone(); x2[:, 4] = torch.randn(32)                  # cambio el último
print("salida", tuple(att(x, x, causal).shape), "· tokens 0–3 iguales:",
      torch.allclose(att(x, x, causal)[:, :4], att(x2, x2, causal)[:, :4]))
'''),
 "ep10_encoder": ("MultiHead", bloque("ENCODER"), '''
capa = CapaEncoder(d=32, h=4, d_ff=128, p=0.1).eval()
x = torch.randn(2, 10, 32)
print("entra", tuple(x.shape), "sale", tuple(capa(x).shape),
      "· parámetros", sum(p.numel() for p in capa.parameters()))
'''),
 "ep10_decoder": ("MultiHead, ffn", bloque("DECODER"), '''
capa = CapaDecoder(d=32, h=4, d_ff=128, p=0.1).eval()
memoria, y = torch.randn(2, 10, 32), torch.randn(2, 7, 32)  # 10 de entrada, 7 escritos
causal = torch.triu(torch.ones(7, 7, dtype=torch.bool), diagonal=1)
print("sale", tuple(capa(y, memoria, causal).shape),
      "· parámetros", sum(p.numel() for p in capa.parameters()))
'''),
 "ep10_greedy": ("Transformer, BOS, datos", bloque("GREEDY"), '''
m = Transformer(); m.load_state_dict(torch.load(CKPT)); m.eval()
x, y, y_in = datos(1000, torch.Generator().manual_seed(123))   # 1000 nuevas
pred = greedy(m, x)
forzado = m(x, y_in).argmax(-1)               # con la respuesta como entrada
for nombre, p in [("greedy", pred), ("teacher forcing", forzado)]:
    print(f"{nombre:15}: secuencias {(p == y).all(1).float().mean():.1%}"
          f" · dígitos {(p == y).float().mean():.1%}")
print("entrada:", *x[0].tolist()); print("salida: ", *pred[0].tolist())
'''),
 "ep10_modelo": ("V, pe, CapaEncoder, CapaDecoder", bloque("MODELO"), '''
m = Transformer()
print(f"parámetros: {sum(p.numel() for p in m.parameters()):,}"
      f" · embeddings: {m.emb.weight.numel()} (una sola vez)")
'''),
}
for nombre, (imp, codigo, prueba) in SNIP.items():
    open(os.path.join(AQUI, nombre + ".py"), "w").write(CAB.format(imp=imp) + "# >>>\n" + codigo + prueba + "# <<<\n")
print("ok", list(SNIP))
