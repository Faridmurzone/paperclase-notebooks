import torch
torch.manual_seed(1)
T, d, d_a = 4, 3, 5
H = torch.randn(T, d)              # h_1..h_4: un estado por palabra de entrada
s = torch.randn(d)                 # s_{i-1}: el estado del decoder
W_a, U_a = torch.randn(d_a, d), torch.randn(d_a, d)
v_a = torch.randn(d_a)
# >>>
# energía: una red chica puntúa cada h_j contra el estado s
e = torch.stack([v_a @ torch.tanh(W_a @ s + U_a @ H[j])
                 for j in range(T)])
alfa = torch.softmax(e, dim=0)     # pesos que suman 1
c = alfa @ H                       # contexto: promedio pesado de los h_j
print("e    =", e.numpy().round(2))
print("alfa =", alfa.numpy().round(2), " suma:", round(alfa.sum().item(), 2))
print("c    =", c.numpy().round(2), " forma:", tuple(c.shape))
# <<<
