# >>>
def f(x):  return x ** 2         # la pérdida: una parábola
def df(x): return 2 * x          # su derivada: la pendiente

x, lr = 3.0, 0.1                 # punto de partida y learning rate
for t in range(1, 21):
    x = x - lr * df(x)           # un paso cuesta abajo
    if t <= 4 or t == 20:
        print(f"paso {t:2}: x = {x:.3f}   f(x) = {f(x):.3f}")
# <<<
