import torch
k = torch.tensor([[1., 0.], [0., 1.], [0., 0.]])  # keys de 3 tokens
def leer(q, V):                                   # una head
    return torch.softmax(q @ k.T, -1) @ V
# >>>
A = torch.tensor([[1., 0.], [0., 1.], [0., 0.]])  # values: frase A
B = torch.tensor([[0., 1.], [1., 0.], [0., 0.]])  # frase B: 0 y 1 al revés

una = torch.tensor([10., 10.])        # 1 head: quiere los tokens 0 y 1
q1, q2 = torch.tensor([10., 0.]), torch.tensor([0., 10.])  # 2 heads
for nombre, V in [("A", A), ("B", B)]:
    o1 = leer(una, V)
    o2 = torch.cat([leer(q1, V), leer(q2, V)])     # concat de heads
    print(f"frase {nombre}: 1 head {o1.numpy().round(2)}"
          f"   2 heads {o2.numpy().round(2)}")
# <<<
