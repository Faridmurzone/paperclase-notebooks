import torch
torch.manual_seed(0)
X, Y = torch.randn(256, 16), torch.randn(256, 1)

# >>>
red = torch.nn.Linear(16, 1)
opt = torch.optim.AdamW(red.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("cpu")                  # escala para float16
with torch.autocast("cpu", dtype=torch.float16):     # el forward en float16
    salida = red(X)
    perdida = ((salida.float() - Y) ** 2).mean()
print("salida:", salida.dtype); print("escala:", scaler.get_scale())
scaler.scale(perdida).backward()                      # gradientes × 65536
print("norma escalada:  ", round(red.weight.grad.norm().item(), 1))
scaler.unscale_(opt)                                  # ÷ 65536, antes del clip
print("norma real:      ", round(red.weight.grad.norm().item(), 4))
torch.nn.utils.clip_grad_norm_(red.parameters(), 1.0)
scaler.step(opt)                                      # saltea si hubo inf
scaler.update()
# <<<
