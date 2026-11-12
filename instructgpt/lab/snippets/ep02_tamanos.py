# Tabla 6 del paper (apéndice A.3): cantidad de prompts por dataset, split y origen.
# >>>
tabla6 = {   # (split, origen): prompts
    "SFT": {("train", "etiquetador"): 11_295, ("train", "cliente"): 1_430,
            ("valid", "etiquetador"): 1_550, ("valid", "cliente"): 103},
    "RM":  {("train", "etiquetador"): 6_623, ("train", "cliente"): 26_584,
            ("valid", "etiquetador"): 3_488, ("valid", "cliente"): 14_399},
    "PPO": {("train", "cliente"): 31_144, ("valid", "cliente"): 16_185},
}
for nombre, t in tabla6.items():
    train = sum(n for (s, o), n in t.items() if s == "train")
    cliente = t.get(("train", "cliente"), 0)
    print(f"{nombre:4} train {train:6,}  de clientes {cliente / train:4.0%}")
# <<<
