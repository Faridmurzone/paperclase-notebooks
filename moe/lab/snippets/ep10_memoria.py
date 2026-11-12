import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _granite import cargar
tok, model = cargar()
# >>>
total = sum(p.numel() for p in model.parameters())
expertos = sum(p.numel() for n, p in model.named_parameters()
               if ".experts." in n)
por_experto = expertos / (24 * 32)            # 24 capas × 32 expertos
activos = total - expertos + 24 * 8 * por_experto   # 8 por token

print(f"totales : {total / 1e6:6.0f} M")
print(f"expertos: {expertos / 1e6:6.0f} M ({expertos / total:.0%})")
print(f"activos : {activos / 1e6:6.0f} M por token")
print(f"bf16    : {2 * total / 1e9:.2f} GB, {2 * activos / 1e9:.2f} por token")
# <<<
