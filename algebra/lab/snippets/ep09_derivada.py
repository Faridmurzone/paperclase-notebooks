import torch
# >>>
def f(x):
    return x ** 2

x0 = 3.0
for h in [1.0, 0.1, 0.001]:              # cociente de diferencias
    print(f"h={h:<6} (f(x+h) - f(x)) / h = {(f(x0 + h) - f(x0)) / h:.4f}")

x = torch.tensor(3.0, requires_grad=True)
y = f(x)
y.backward()                              # autograd: dy/dx
print("autograd: f'(3) =", x.grad.item())

d = 0.2                                   # aproximación lineal
print(f"f(3,2) = {f(3.2):.2f}   f(3) + f'(3)*0,2 = {f(3) + 6 * d:.2f}")
# <<<
