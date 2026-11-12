import torch
torch.manual_seed(0)
d_x, d_h, T = 3, 4, 5                  # entrada de 3, estado de 4, 5 pasos
W_xh = torch.randn(d_h, d_x) * 0.5
W_hh = torch.randn(d_h, d_h) * 0.5
b = torch.zeros(d_h)
xs = torch.randn(T, d_x)               # una secuencia de 5 vectores
# >>>
h = torch.zeros(d_h)                   # h_0: el estado arranca en cero
for t in range(T):                     # un paso por token, en orden
    h = torch.tanh(W_xh @ xs[t] + W_hh @ h + b)
    print(f"t={t + 1}  h = {h.numpy().round(2)}")

rnn = torch.nn.RNN(d_x, d_h, bias=False)   # la misma cuenta, en PyTorch
rnn.weight_ih_l0.data, rnn.weight_hh_l0.data = W_xh, W_hh
salidas, h_final = rnn(xs.unsqueeze(1))
print("igual a nn.RNN:", torch.allclose(h, h_final.flatten(), atol=1e-6))
# <<<
