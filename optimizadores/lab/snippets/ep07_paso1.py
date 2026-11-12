import torch
r = lambda x, d=4: [round(v, d) for v in x.tolist()]

# >>>
th = torch.tensor([1.0, -2.0])                  # θ: dos pesos
g = torch.tensor([0.5, -4.0])                   # g: su gradiente en t = 1
eta, lam, b1, b2, eps, t = 0.1, 0.1, 0.9, 0.999, 1e-8, 1

th = th * (1 - eta * lam)                       # 1. weight decay
print("1. decay:", r(th))
m = b1 * 0 + (1 - b1) * g                       # 2. primer momento
v = b2 * 0 + (1 - b2) * g ** 2                  # 3. segundo momento
print("2-3. m:", r(m), " v:", r(v, 5))
mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)  # 4. corrección de sesgo
print("4. m̂:", r(mh), " v̂:", r(vh))
th = th - eta * mh / (vh.sqrt() + eps)          # 5. el paso
print("5. θ:", r(th))
# <<<
