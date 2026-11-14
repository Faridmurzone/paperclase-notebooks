import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _clip import datos, frase, PLANTILLAS, batch_un_digito_por_clase
# >>>
Xtr, ytr, Xte, yte = datos()          # load_digits: 8×8 = 64 píxeles
print("imágenes:", tuple(Xtr.shape), "train,", len(yte), "test")
for p in PLANTILLAS:                  # 4 plantillas de texto
    print(f"  {p!r:26} → {frase(7, p)!r}")

rng = random.Random(0)
imgs, textos = batch_un_digito_por_clase(Xtr, ytr, rng)
print("batch:", tuple(imgs.shape), textos[:3])
# <<<
