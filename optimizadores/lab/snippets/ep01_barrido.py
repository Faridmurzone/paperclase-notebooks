import json, os
DATOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datos")

# >>>
def descenso(lr, x=3.0, pasos=20):
    xs = [x]
    for _ in range(pasos):
        x = x - lr * 2 * x       # f(x) = x², f'(x) = 2x
        xs.append(x)
    return xs

for lr in [0.01, 0.1, 0.5, 0.9, 1.0, 1.1]:
    xs = descenso(lr)
    print(f"lr {lr:4}: factor {1 - 2*lr:5.2f}  x final {xs[-1]:8.3f}")
# <<<
json.dump({str(lr): descenso(lr) for lr in [0.01, 0.1, 0.5, 0.9, 1.0, 1.1]},
          open(os.path.join(DATOS, "ep01_barrido.json"), "w"))
