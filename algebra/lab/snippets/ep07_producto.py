import torch
torch.manual_seed(0)
# >>>
# probabilidad de cada token de una respuesta de 300 tokens
p = torch.rand(300) * 0.6 + 0.2          # entre 0,2 y 0,8

producto = torch.prod(p)                 # P(respuesta) directo
suma_logs = torch.log(p).sum()           # log P(respuesta)

print("producto de 300 probs:", producto.item())
print("suma de 300 logs:", round(suma_logs.item(), 1))
print("en float64:", torch.prod(p.double()).item())
print("e^(suma de logs):", torch.exp(suma_logs.double()).item())
# <<<
