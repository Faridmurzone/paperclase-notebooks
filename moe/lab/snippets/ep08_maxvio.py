import torch
# >>>
def maxvio(carga):                      # (máx_i c_i − c̄) / c̄
    c = carga.float()
    return ((c.max() - c.mean()) / c.mean()).item()

ejemplos = {"pareja":    [64, 64, 64, 64, 64, 64, 64, 64],
            "despareja": [90, 40, 70, 64, 30, 80, 60, 78],
            "colapso":   [400, 40, 30, 20, 10, 6, 4, 2]}
for nombre, c in ejemplos.items():
    print(f"{nombre:10} suma {sum(c)} · MaxVio {maxvio(torch.tensor(c)):.2f}")
# <<<
