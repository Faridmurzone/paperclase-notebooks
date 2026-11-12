import torch
torch.set_printoptions(precision=2, sci_mode=False)
# >>>
# 3 tokens, d_k = 2: números redondos para seguir a mano
Q = torch.tensor([[2., 0.], [0., 2.], [1., 1.]])
K = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
V = torch.tensor([[10., 0.], [0., 10.], [5., 5.]])

S = Q @ K.T / 2 ** 0.5       # [3, 3]: q_i · k_j / √d_k
A = torch.softmax(S, dim=-1) # cada fila suma 1
out = A @ V                  # [3, 2]: promedio pesado de V
print("S =\n", S)
print("A =\n", A)
print("out =\n", out)
# <<<
