import json, os
hist = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos", "ep03_sft.json")))
# >>>
# hist: (epoch, loss de train, loss de validación, métrica) por epoch
por_loss = min(hist, key=lambda h: h[2])       # la regla de siempre
por_metrica = max(hist, key=lambda h: h[3])    # la regla del paper
for nombre, (ep, _, l_va, acc) in [("por val loss", por_loss),
                                   ("por métrica", por_metrica)]:
    print(f"{nombre:13} → epoch {ep:2d}  val {l_va:.3f}  ordena {acc:.0%}")
# <<<
