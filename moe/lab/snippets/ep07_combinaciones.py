from math import comb
# >>>
casos = [("16 expertos, top-2", 16, 2),
         ("partidos en 4: top-8", 64, 8),
         ("Granite", 32, 8),
         ("DeepSeek-V3", 256, 8)]
for nombre, N, K in casos:
    print(f"{nombre:21} C({N},{K}) = {comb(N, K):.3g}")
# <<<
