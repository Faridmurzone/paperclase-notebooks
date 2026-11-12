# Números de la Tabla 14 de InstructGPT (arXiv 2203.02155), 175B, zero-shot. Columnas: GPT, SFT, PPO, PPO-ptx.
# >>>
tabla14 = {"DROP (F1)":      [27.53, 15.79, 13.08, 15.23],
           "SQuADv2 (F1)":   [64.30, 57.67, 43.68, 59.85],
           "HellaSwag (acc)": [0.781, 0.753, 0.743, 0.807],
           "FR→EN (BLEU)":   [38.92, 36.90, 24.16, 34.28]}
for nombre, (gpt, sft, ppo, ptx) in tabla14.items():
    impuesto = (gpt - ppo) / gpt             # cuánto pierde PPO, relativo a GPT-3
    recupera = (ptx - ppo) / (gpt - ppo)     # qué parte devuelve la mezcla
    print(f"{nombre:16} PPO {-impuesto:+.0%}   ptx recupera {recupera:.0%}")
# <<<
