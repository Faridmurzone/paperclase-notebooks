import json, os
c = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep01_conteo.json")))
N = c["total"]
# >>>
r, capas = 8, 26
q = r * (2048 + 2304)                       # B y A de q_proj
v = r * (1024 + 2304)                       # B y A de v_proj
Ne = capas * (q + v)                        # lo único que se entrena
print(f"entrenables: {Ne:,} ({Ne / N:.3%})")

GB = 1e9
completo = 16 * N                           # 4 copias de 4 bytes
lora = 4 * N + 16 * Ne                      # W solo pesos; A y B, las 4
print(f"ajuste completo: {completo / GB:5.1f} GB")
print(f"con LoRA:        {lora / GB:5.1f} GB")
print(f"archivo por tarea: {4 * Ne / 1e6:.1f} MB contra {4 * N / GB:.1f} GB")
# <<<
