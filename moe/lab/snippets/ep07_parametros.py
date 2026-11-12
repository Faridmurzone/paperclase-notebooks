import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _juguete import MoE, parametros
import inspect
# >>>
def activos(moe):                       # router + k ruteados + compartidos
    return (moe.router.weight.numel() + moe.k * parametros(moe.expertos[0])
            + sum(parametros(c) for c in moe.compartidos))

configs = {"grueso": MoE(32, 64, E=8, k=2),        # 8 de h=64, top-2
           "fino": MoE(32, 16, E=32, k=8),          # 32 de h=16, top-8
           "fino+comp": MoE(32, 16, E=31, k=7, n_compartidos=1)}
print("           total  activos")
for nombre, moe in configs.items():
    print(f"{nombre:9} {parametros(moe):6d}  {activos(moe):7d}")
fuente = inspect.getsource(MoE.forward).splitlines()   # el camino fijo
i = next(j for j, l in enumerate(fuente) if "self.compartidos" in l)
print("\n".join(l.strip() for l in fuente[i:i + 2]))
# <<<
