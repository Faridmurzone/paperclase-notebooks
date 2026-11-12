import torch
torch.manual_seed(0)
# >>>
capa = torch.nn.Linear(3, 2)                # 6 pesos + 2 sesgos
opt = torch.optim.Adam(capa.parameters(), lr=0.1)

x = torch.tensor([[1.0, 2.0, 3.0]])
perdida = capa(x).pow(2).sum()
perdida.backward()                          # llena .grad
w0 = capa.weight.detach().clone()
opt.step()                                  # crea m y v

w = capa.weight
estado = opt.state[w]
print("peso      ", tuple(w.shape))
print("gradiente ", tuple(w.grad.shape))
print("m (exp_avg)   ", tuple(estado["exp_avg"].shape))
print("v (exp_avg_sq)", tuple(estado["exp_avg_sq"].shape))
print("bytes por número:", w.element_size())
print("cambio:", [round(c, 3) for c in (w.detach() - w0).flatten().tolist()])
# <<<
