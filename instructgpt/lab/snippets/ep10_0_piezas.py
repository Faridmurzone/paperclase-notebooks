# Qué modelos hay que tener en memoria durante el entrenamiento de preferencias, según cada receta.
# >>>
recetas = {
    "InstructGPT (PPO-ptx)": ["política", "SFT (referencia)", "RM", "value function"],
    "DPO": ["política", "SFT (referencia)"],
    "CAI (RL-CAI)": ["política", "referencia", "PM", "value function"],
}
generan_en_el_loop = {"InstructGPT (PPO-ptx)": True, "DPO": False, "CAI (RL-CAI)": True}
for receta, modelos in recetas.items():
    online = "muestrea en cada paso" if generan_en_el_loop[receta] else "pares fijos, sin muestrear"
    print(f"{receta:22} {len(modelos)} modelos · {online}")
# <<<
