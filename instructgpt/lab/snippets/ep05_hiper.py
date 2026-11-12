# Apéndice C.4 del paper.
# >>>
episodios = 256_000          # respuestas generadas en todo el entrenamiento
batch, minibatch = 512, 64
beta, gamma = 0.02, 27.8     # freno de KL y peso del preentrenamiento
ptx_por_episodio = 8         # 8 veces más ejemplos de preentrenamiento

iteraciones = episodios // batch
print("iteraciones:", iteraciones)
print("minibatches por iteración:", batch // minibatch)
print("ejemplos de ptx:", f"{ptx_por_episodio * episodios:,}")
print("parejo: 64 respuestas +", ptx_por_episodio * minibatch, "ptx")
print("1 pasada | clip 0,2 | EMA 0,992")
# <<<
