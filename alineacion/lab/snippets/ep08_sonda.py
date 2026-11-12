import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0)
torch.set_grad_enabled(False)
tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", dtype=torch.float32)
P = [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
     ("Japan", "Tokyo"), ("China", "Beijing"), ("Russia", "Moscow"), ("Egypt", "Cairo"),
     ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Austria", "Vienna"), ("Poland", "Warsaw"),
     ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Finland", "Helsinki"), ("Denmark", "Copenhagen"),
     ("Ireland", "Dublin"), ("Hungary", "Budapest"), ("Kenya", "Nairobi"), ("Peru", "Lima"),
     ("Chile", "Santiago"), ("Argentina", "Buenos Aires"), ("Thailand", "Bangkok"), ("Cuba", "Havana"),
     ("Turkey", "Ankara"), ("Iran", "Tehran"), ("Canada", "Ottawa"), ("Australia", "Canberra"),
     ("Belgium", "Brussels"), ("Vietnam", "Hanoi")]
frases, y = [], []
for i, (pais, cap) in enumerate(P):
    frases += [f"The capital of {pais} is {cap}.", f"The capital of {pais} is {P[(i + 7) % len(P)][1]}."]
    y += [1, 0]
y = torch.tensor(y)
# >>>
def activacion(frase, capa=12):               # residual stream, último token
    cache = {}
    h = model.model.layers[capa].register_forward_hook(
        lambda m, i, o: cache.update(x=o[0, -1]))
    model(**tok(frase, return_tensors="pt")); h.remove()
    return cache["x"]

X = torch.stack([activacion(f) for f in frases])        # [60, 2304]
tr, te = slice(0, 40), slice(40, 60)                    # 20 países / 10 nuevos

probe = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
print("logística, acc test:", probe.score(X[te], y[te]))

r = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)   # μ+ − μ−
umbral = (X[tr] @ r).median()
acc = ((X[te] @ r > umbral) == y[te]).float().mean()
print("dif. de medias, acc test:", acc.item())
w = torch.tensor(probe.coef_[0], dtype=torch.float32)
print(f"coseno(w, r) = {torch.cosine_similarity(w, r, dim=0):.2f}")
# <<<
