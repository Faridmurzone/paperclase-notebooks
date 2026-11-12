import torch
def r(t, d=3): return [round(x, d) for x in t.tolist()]
# >>>
logits = torch.tensor([2.0, 1.0, 0.1])   # 3 tokens posibles

e = torch.exp(logits)                    # 1) exponencial: todo > 0
p = e / e.sum()                          # 2) dividir por la suma
print("e^logits:", r(e, 2))
print("softmax: ", r(p), " suma:", round(p.sum().item(), 4))
print("torch:   ", r(torch.softmax(logits, dim=0)))

p100 = torch.softmax(logits + 100, dim=0)  # sumarle 100 a todos
print("logits + 100:", r(p100))

z = 1.5                                    # dos opciones: (z, 0)
print("softmax([z, 0])[0] =", round(torch.softmax(torch.tensor([z, 0.0]), 0)[0].item(), 4))
print("sigmoid(z)        =", round(torch.sigmoid(torch.tensor(z)).item(), 4))
# <<<
