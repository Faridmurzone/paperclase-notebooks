import torch
torch.manual_seed(0)
d = 32
# >>>
rnn = torch.nn.RNN(d, d, batch_first=True)    # init por defecto
x = torch.randn(1, 41, d, requires_grad=True) # 41 tokens
salidas, _ = rnn(x)
salidas[0, -1].sum().backward()     # la salida del último token...

g = x.grad[0].norm(dim=1)           # ...¿cuánto depende de cada x_t?
for dist in [0, 5, 10, 20, 30, 40]:
    print(f"distancia {dist:2}: |∂h_T/∂x_t| = {g[40 - dist]:.1e}")
# <<<
