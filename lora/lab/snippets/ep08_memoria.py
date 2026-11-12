total = 2_614_341_888                         # parámetros de Gemma 2 2B (ep. 6)
# >>>
GB = 8e9                                      # bits en un gigabyte
bits = {
    "float32": 32,
    "bfloat16": 16,
    "4 + absmax": 4 + 32 / 64,
    "4 + doble": 4 + 8 / 64 + 32 / (64 * 256),
}
for nombre, b in bits.items():
    print(f"{nombre:11} {b:6.3f} bits {total * b / GB:6.2f} GB")
# <<<
